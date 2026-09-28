# (C) Copyright 2024 Anemoi contributors.
# Copyright (C) Bull S.A.S - 2025
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hydra.utils import get_class
from peft import PeftModel
from peft import get_peft_model

from anemoi.training.train.methods.single import SingleTraining

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema
    from training.src.anemoi.training.tasks.base import BaseTask


LOGGER = logging.getLogger(__name__)


class PeftSingleTraining(SingleTraining):
    """PEFT training method."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        task: BaseTask,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:

        super().__init__(
            config=config,
            task=task,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        peft_config_class = get_class(config.training.peft_config._target_)
        self.peft_config = peft_config_class(**config.training.peft_config.config)

    def on_load_checkpoint(self, checkpoint: torch.nn.Module) -> None:
        self._update_checkpoint_state_dict_for_load(checkpoint)

        self._ckpt_model_name_to_index = {
            dataset_name: data_indices.name_to_index
            for dataset_name, data_indices in checkpoint["hyper_parameters"]["data_indices"].items()
        }
        if "PeftSingleTraining" in checkpoint["hyper_parameters"]["config"].training.training_method:
            self._inject_peft_adapters()

    def on_checkpoint_loaded(self) -> None:
        if not isinstance(self.model, PeftModel):
            self._inject_peft_adapters()

    def _inject_peft_adapters(self) -> None:
        get_peft_model(self.model, self.peft_config)
        LOGGER.info("PEFT adapters injected into the model")
