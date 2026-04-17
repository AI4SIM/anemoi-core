# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.mse import MSELoss

LOGGER = logging.getLogger(__name__)


class WeightedCharbonnierLoss(MSELoss):
    """Weighted Charbonnier loss for use with diffusion models.

    This loss applies weights to the Charbonnier difference
    """

    name: str = "weighted_charbonnier"

    def __init__(
        self,
        epsilon: float = 1e-6,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the WeightedCharbonnierLoss.

        Parameters
        ----------
        epsilon : float, optional
            Small constant to avoid division by zero, by default 1e-6
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: str = "avg",
    ) -> torch.Tensor:
        """Calculates the weighted Charbonnier loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        weights : torch.Tensor | None, optional
            Weights to apply to the Charbonnier loss, by default None
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Weighted Charbonnier loss
        """
        is_sharded = grid_shard_slice is not None
        out = self.calculate_difference(pred, target)
        out = torch.sqrt(out + self.epsilon**2)

        if weights is not None:
            out = out * weights

        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        return self.reduce(out, squash, group=group if is_sharded else None, squash_mode=squash_mode)
