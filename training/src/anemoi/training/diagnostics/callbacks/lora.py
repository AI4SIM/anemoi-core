# (C) Copyright 2024 Anemoi contributors.
# (C) Copyright 2025 BULL SAS.
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

from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


class LoRAAdapters(Callback):
    """Inject LoRA adapters in a pre-trained model for fine-tuning."""

    def __init__(self, config: OmegaConf, **kwargs) -> None:
        """Initialize LoRA injection callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings
        kwargs : dict
            Keyword arguments for LoRA configuration, such as `r`, `lora_alpha`, `lora_dropout`, etc.
            See `peft.LoraConfig` for details.
        """
        super().__init__()
        self.config = config
        self.lora_config = LoraConfig(**kwargs, task_type=TaskType.FEATURE_EXTRACTION)

    def on_fit_start(self, _: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the training data.

        Parameters
        ----------
        _ : pl.Trainer
            Not used
        pl_module : pl.LightningModule
            Pytorch Lightning module
        """
        get_peft_model(pl_module.model, self.lora_config)
