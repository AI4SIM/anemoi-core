# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Variogram Score loss.

The variogram score of order p (Scheuerer & Hamill 2015) is a proper scoring
rule for multivariate probabilistic forecasts that evaluates whether the
predicted dependence structure matches the observation:

    VS_p(F, y) = sum_{i,j} w_ij * ( E_F[|X_i - X_j|^p] - |y_i - y_j|^p )^2

where X_i and X_j are the ith and jth components of a random vector X
distributed according to F, y is the observation vector, and w_ij are
non-negative weights. For an M-member ensemble forecast {X^(1), ..., X^(M)},
the expected variogram is estimated as:

    E_F[|X_i - X_j|^p] ≈ (1/M) * sum_{k=1}^{M} |X_i^(k) - X_j^(k)|^p

The score targets the dependence structure (not marginals) and is insensitive
to a uniform bias. Recommended values: p=0.5 for non-Gaussian fields,
p=1.0 for robustness, p=2.0 for spectral-equivalent form.

Two modes are supported:

- **ensemble** (classical multivariate scoring rule): pred has M > 1 ensemble
  members. The components i, j of the random vector X are the predicted
  variables. The expected variogram E_F[|X_i - X_j|^p] is estimated by
  averaging per-member variable increments. Since the number of variables
  is limited, all variable pairs are computed without sampling.

- **deterministic**: pred has no ensemble dimension (M = 1). The components
  i, j are spatial indices. The forecast variogram reduces to
  |X_i - X_j|^p from the single prediction. Spatial pairs are
  stochastically sampled to avoid O(d^2) cost on large grids.

For large grids in deterministic mode, spatial pairs can be stratified by lag
distance on structured (2D) grids to ensure all spatial scales are represented.

References
----------
Scheuerer, M. and T. M. Hamill, 2015:
    "Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of
    Multivariate Quantities." Mon. Wea. Rev., 143, 1321-1334.
    https://doi.org/10.1175/MWR-D-14-00269.1

Pic, R., C. Dombry, P. Naveau, and M. Taillardat, 2025:
    "Proper scoring rules for multivariate probabilistic forecasts based on
    aggregation and transformation." Adv. Stat. Clim. Meteorol. Oceanogr.,
    11, 23-58. https://doi.org/10.5194/ascmo-11-23-2025

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Literal

import torch

from anemoi.training.losses.base import BaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class VariogramScore(BaseLoss):
    """Variogram score of order p (Scheuerer & Hamill 2015).

    A proper scoring rule for multivariate probabilistic forecasts.
    Evaluates whether the predicted dependence structure matches the
    observation by comparing pairwise increments |X_i - X_j|^p between
    components of the forecast and observation vectors.

    Modes
    -----
    - "ensemble": pred has shape (bs, t, ens, latlon, v) with ens > 1.
      Target has shape (bs, t, latlon, v). The components i, j of the
      d-variate random vector X are the variables. At each grid point, the
      VS evaluates whether the ensemble correctly captures the inter-variable
      dependence. All variable pairs are used (no sampling needed since the
      number of variables is limited).

    - "deterministic": pred has shape (bs, t, latlon, v) (no ensemble dim).
      Target has shape (bs, t, latlon, v). The components i, j are spatial
      indices. The VS evaluates the spatial structure of the prediction for
      each variable. Spatial pairs are stochastically sampled to avoid
      O(d^2) cost on large grids.
    """

    def __init__(
        self,
        mode: Literal["ensemble", "deterministic"] = "ensemble",
        p: float = 0.5,
        n_pairs: int = 50_000,
        n_bins: int = 20,
        max_lag_fraction: float = 0.5,
        resample_every_step: bool = True,
        eps: float = 1e-6,
        ignore_nans: bool = False,
    ) -> None:
        """Initialize VariogramScore.

        Parameters
        ----------
        mode : {"ensemble", "deterministic"}
            - "ensemble": pred has ensemble dimension, target does not.
              Classical VS for multivariate probabilistic forecasts.
              Pairs are over the variable dimension (all pairs used).
            - "deterministic": neither pred nor target has ensemble dimension.
              VS for deterministic models evaluating spatial structure.
              Pairs are over the spatial dimension (stochastically sampled).
        p : float
            Order of the variogram. p=0.5 (default) captures non-Gaussian
            structure; p=1.0 is robust; p=2.0 is equivalent to a spectral loss.
        n_pairs : int
            Number of spatial pairs to sample per forward pass.
            Only used in deterministic mode.
        n_bins : int
            Number of lag bins for stratified sampling (when grid_shape provided).
            Only used in deterministic mode.
        max_lag_fraction : float
            Maximum lag as a fraction of the grid side length.
            E.g., 0.5 means pairs up to half the domain size.
            Only used in deterministic mode.
        resample_every_step : bool
            If True (default), draw new random pairs each forward pass.
            If False, fix pairs at first call (deterministic evaluation).
            Only used in deterministic mode.
        eps : float
            Small constant for numerical stability in |x|^p gradient at x=0.
        ignore_nans : bool
            Allow nans in the loss computation.
        """
        super().__init__(ignore_nans=ignore_nans)

        self.mode = mode
        self.p = p
        self.n_pairs = n_pairs
        self.n_bins = n_bins
        self.max_lag_fraction = max_lag_fraction
        self.resample_every_step = resample_every_step
        self.eps = eps

        # Cache for fixed spatial pairs (resample_every_step=False, deterministic mode)
        self._cached_pairs: tuple[torch.Tensor, torch.Tensor] | None = None
        # Cache for all variable pairs (ensemble mode)
        self._var_pairs_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self.supports_sharding = False

    @property
    def name(self) -> str:
        return f"variogram_score_{self.mode}_p{self.p}"

    # ------------------------------------------------------------------
    # Pair generation
    # ------------------------------------------------------------------

    def _generate_pairs_uniform(
        self,
        n_grid: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate uniform random spatial pairs (for unstructured grids).

        Parameters
        ----------
        n_grid : int
            Number of grid points.
        device : torch.device
            Device for output tensors.

        Returns
        -------
        idx_a, idx_b : torch.Tensor, shape (n_pairs,)
            Index pairs into the spatial dimension.
        """
        idx_a = torch.randint(0, n_grid, (self.n_pairs,), device=device)
        idx_b = torch.randint(0, n_grid, (self.n_pairs,), device=device)

        # Avoid self-pairs
        same = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % n_grid

        return idx_a, idx_b

    def _generate_pairs_stratified(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate lag-stratified random spatial pairs on a 2D grid.

        Pairs are stratified by lag distance to ensure all spatial scales
        are represented in the variogram estimate.

        Parameters
        ----------
        height : int
            Grid height.
        width : int
            Grid width.
        device : torch.device
            Device for output tensors.

        Returns
        -------
        idx_a, idx_b : torch.Tensor, shape (n_pairs_actual,)
            Index pairs (flattened 2D indices).
        """
        pairs_per_bin = self.n_pairs // self.n_bins
        max_lag = int(self.max_lag_fraction * min(height, width))

        bin_edges = torch.linspace(1, max_lag, self.n_bins + 1, device=device)

        all_idx_a = []
        all_idx_b = []

        for b in range(self.n_bins):
            r_lo = bin_edges[b].item()
            r_hi = bin_edges[b + 1].item()

            # Over-sample then filter by distance (rejection sampling)
            oversample = pairs_per_bin * 4
            r_hi_int = int(r_hi) + 1

            di = torch.randint(-r_hi_int, r_hi_int + 1, (oversample,), device=device)
            dj = torch.randint(-r_hi_int, r_hi_int + 1, (oversample,), device=device)
            dist = (di.float().square() + dj.float().square()).sqrt()
            valid = (dist >= r_lo) & (dist < r_hi) & ((di != 0) | (dj != 0))

            di = di[valid][:pairs_per_bin]
            dj = dj[valid][:pairs_per_bin]
            n_valid = di.shape[0]

            if n_valid == 0:
                continue

            # Random anchors
            i0 = torch.randint(0, height, (n_valid,), device=device)
            j0 = torch.randint(0, width, (n_valid,), device=device)
            i1 = (i0 + di).clamp(0, height - 1)
            j1 = (j0 + dj).clamp(0, width - 1)

            # Flatten to linear indices
            all_idx_a.append(i0 * width + j0)
            all_idx_b.append(i1 * width + j1)

        idx_a = torch.cat(all_idx_a)
        idx_b = torch.cat(all_idx_b)

        return idx_a, idx_b

    def _generate_all_pairs(
        self,
        n: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate all pairs (i, j) with i < j.

        Used in ensemble mode for variable pairs.

        Parameters
        ----------
        n : int
            Number of elements (variables).
        device : torch.device
            Device for output tensors.

        Returns
        -------
        idx_a, idx_b : torch.Tensor, shape (n*(n-1)/2,)
            Index pairs.
        """
        if self._var_pairs_cache is not None:
            cached_a, _ = self._var_pairs_cache
            if cached_a.shape[0] == n * (n - 1) // 2 and cached_a.device == device:
                return self._var_pairs_cache

        idx_a, idx_b = torch.triu_indices(n, n, offset=1, device=device)
        self._var_pairs_cache = (idx_a, idx_b)
        return idx_a, idx_b

    def _get_pairs(
        self,
        n_grid: int,
        device: torch.device,
        grid_shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or generate spatial pairs.

        In deterministic mode, pairs are stochastically sampled.
        """
        if self._cached_pairs is not None and not self.resample_every_step:
            return self._cached_pairs

        if grid_shape is not None and len(grid_shape) == 2:
            pairs = self._generate_pairs_stratified(grid_shape[0], grid_shape[1], device)
        else:
            pairs = self._generate_pairs_uniform(n_grid, device)

        if not self.resample_every_step:
            self._cached_pairs = pairs

        return pairs

    # ------------------------------------------------------------------
    # Core variogram score computation
    # ------------------------------------------------------------------

    def _variogram_score_ensemble(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the variogram score over variable pairs (ensemble mode).

        VS_p = mean_{pairs (i,j)} ( E_F[|X_i - X_j|^p] - |y_i - y_j|^p )^2

        where i, j index variables (components of the random vector X).
        E_F[|X_i - X_j|^p] ≈ (1/M) sum_{k=1}^M |X_i^(k) - X_j^(k)|^p

        Parameters
        ----------
        pred : torch.Tensor
            Predicted field, shape (bs, t, ens, latlon, v).
        target : torch.Tensor
            Ground truth, shape (bs, t, latlon, v).
        idx_a : torch.Tensor
            First variable indices, shape (n_var_pairs,).
        idx_b : torch.Tensor
            Second variable indices, shape (n_var_pairs,).

        Returns
        -------
        torch.Tensor
            Per-grid-point variogram score, shape (bs, t, latlon).
        """
        # Observation variogram: |y_i - y_j|^p for each variable pair
        # target: (bs, t, latlon, v) → gather variable pairs → (bs, t, latlon, n_pairs)
        tgt_a = target[:, :, :, idx_a]
        tgt_b = target[:, :, :, idx_b]
        obs_variogram = (tgt_a - tgt_b).abs().clamp(min=self.eps).pow(self.p)

        # Forecast variogram: (1/M) sum_k |X_i^(k) - X_j^(k)|^p
        # pred: (bs, t, ens, latlon, v) → gather variable pairs → (bs, t, ens, latlon, n_pairs)
        pred_a = pred[:, :, :, :, idx_a]
        pred_b = pred[:, :, :, :, idx_b]
        # Per-member increments, averaged over ensemble dim (dim=2)
        fcst_variogram = (
            (pred_a - pred_b).abs().clamp(min=self.eps).pow(self.p).mean(dim=2)
        )  # fcst_variogram: (bs, t, latlon, n_pairs)

        # Variogram score: squared difference, averaged over variable pairs
        return (fcst_variogram - obs_variogram).square().mean(dim=-1)  # (bs, t, latlon)

    def _variogram_score_deterministic(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the variogram score over spatial pairs (deterministic mode).

        VS_p = mean_{pairs (i,j)} ( |pred_i - pred_j|^p - |y_i - y_j|^p )^2

        where i, j index spatial locations.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted field, shape (bs, t, latlon, v).
        target : torch.Tensor
            Ground truth, shape (bs, t, latlon, v).
        idx_a : torch.Tensor
            First spatial indices, shape (n_pairs,).
        idx_b : torch.Tensor
            Second spatial indices, shape (n_pairs,).

        Returns
        -------
        torch.Tensor
            Per-variable variogram score, shape (bs, t, v).
        """
        # Observation variogram: |y_i - y_j|^p for each spatial pair
        # target: (bs, t, latlon, v) → gather spatial pairs → (bs, t, n_pairs, v)
        tgt_a = target[:, :, idx_a, :]
        tgt_b = target[:, :, idx_b, :]
        obs_variogram = (tgt_a - tgt_b).abs().clamp(min=self.eps).pow(self.p)

        # Forecast variogram: |pred_i - pred_j|^p (single prediction)
        # pred: (bs, t, latlon, v) → gather spatial pairs → (bs, t, n_pairs, v)
        pred_a = pred[:, :, idx_a, :]
        pred_b = pred[:, :, idx_b, :]
        fcst_variogram = (pred_a - pred_b).abs().clamp(min=self.eps).pow(self.p)

        # Variogram score: squared difference, averaged over spatial pairs
        return (fcst_variogram - obs_variogram).square().mean(dim=2)  # (bs, t, v)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: str = "avg",
        grid_shape: tuple[int, ...] | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the variogram score loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor.
            - mode="ensemble": shape (bs, time, ensemble, latlon, variable)
            - mode="deterministic": shape (bs, time, latlon, variable)
        target : torch.Tensor
            Target tensor, shape (bs, time, latlon, variable).
            Single realization (no ensemble dimension).
        squash : bool
            Average over variable dimension.
        scaler_indices : tuple[int, ...] | None
            Indices to subset the scaler.
        without_scalers : list[str] | list[int] | None
            Scalers to exclude.
        grid_shard_slice : slice | None
            Not supported (spatial operations need full grid).
        group : ProcessGroup | None
            Distributed group.
        squash_mode : str
            Reduction mode for variable dimension.
        grid_shape : tuple[int, ...] | None
            If provided as (H, W), enables lag-stratified pair sampling
            in deterministic mode. Otherwise uses uniform random pair sampling.
            Ignored in ensemble mode.

        Returns
        -------
        torch.Tensor
            Variogram score loss.
        """
        if grid_shard_slice is not None:
            msg = "VariogramScore does not support grid sharding."
            raise NotImplementedError(msg)

        if self.mode == "ensemble":
            # Pairs over variables: all V*(V-1)/2 variable pairs
            n_vars = pred.shape[-1]
            idx_a, idx_b = self._generate_all_pairs(n_vars, pred.device)

            # Compute VS over variable pairs → shape (bs, t, latlon)
            vs = self._variogram_score_ensemble(pred, target, idx_a, idx_b)

            # Reshape to (bs, t, 1, latlon, 1) for scale/reduce compatibility
            vs = vs[:, :, None, :, None]
        else:
            # Pairs over spatial locations: stochastically sampled
            n_grid = pred.shape[-2]
            idx_a, idx_b = self._get_pairs(n_grid, pred.device, grid_shape)

            # Compute VS over spatial pairs → shape (bs, t, v)
            vs = self._variogram_score_deterministic(pred, target, idx_a, idx_b)

            # Reshape to (bs, t, 1, 1, v) for scale/reduce compatibility
            vs = vs[:, :, None, None, :]

        vs = self.scale(
            vs,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(vs, squash=squash, group=group, squash_mode=squash_mode)
