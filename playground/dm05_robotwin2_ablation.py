"""Controlled RoboTwin S0--S4 training (shared augmentation and hyperparameters)."""

from dataclasses import dataclass, field
from typing import Literal

import tyro

from opendm.constants.robot import ActionMode
from opendm.dataset.robotwin2_ablation import DATA_ROOT, RELATIVE_MODES
from opendm.exp.dm05_exp import DM05DataConfig as BaseDataConfig, DM05Exp as BaseExp
from opendm.exp.dm05_exp import (
    DM05ModelConfig,
    DM05OptimizerConfig as BaseOptimizerConfig,
    DM05TrainerConfig as BaseTrainerConfig,
)


@dataclass
class DM05OptimizerConfig(BaseOptimizerConfig):
    base_lr: float = 4e-5
    optim: Literal["adamw", "muon_adamw"] = "muon_adamw"


@dataclass
class DM05TrainerConfig(BaseTrainerConfig):
    output_dir: str = "user_checkpoints/robotwin2_ablation/s0"
    # 8 GPUs x 2 samples x 8 accumulation steps = global batch 128.
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    save_steps: int = 10000
    num_train_steps: int = 60000
    save_only_model: bool = False


@dataclass
class DM05DataConfig(BaseDataConfig):
    dataset_name: str = "robotwin2_ablation_s0"
    norm_stats_root: str = f"{DATA_ROOT}/norm_stats"

    def _dataset_info(self):
        rung = self.dataset_name.removeprefix("robotwin2_ablation_")
        if rung not in RELATIVE_MODES:
            raise ValueError(f"Unknown ablation dataset: {self.dataset_name}")
        if self.relative_mode != RELATIVE_MODES[rung]:
            raise ValueError(f"{rung} requires relative_mode={RELATIVE_MODES[rung]}")
        if self.action_mode != ActionMode.RELATIVE or not self.add_state:
            raise ValueError(
                "All ablation groups require relative actions and add_state=True"
            )
        return super()._dataset_info()


@dataclass
class DM05Exp(BaseExp):
    use_lora: bool | None = False
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task != "train":
        raise ValueError(
            "This entry trains representations; RoboTwin EEF rollout needs a benchmark adapter"
        )
    exp.train()
