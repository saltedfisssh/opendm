"""Small, single-process configuration for debugging RoboChallenge SFT."""

from dataclasses import dataclass, field

import tyro

from opendm.exp.dm05_exp import (
    DM05DataConfig as _DM05DataConfig,
)
from opendm.exp.dm05_exp import (
    DM05Exp as _DM05Exp,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)


@dataclass
class DM05DebugDataConfig(_DM05DataConfig):
    """Use ARX5 multi-task data and keep initial norm-stat work short."""

    dataset_name: str = field(default="table30v2_arx5")
    compute_norm_stats_max_batches: int | None = field(default=1)
    norm_stats_root: str = field(default="./norm_stats/robochallenge_debug")

    def compute_norm_stats(
        self,
        action_horizon: int,
        batch_size: int = 8,
        num_workers: int = 0,
        balance_tasks: bool = True,
    ) -> None:
        super().compute_norm_stats(
            action_horizon=action_horizon,
            batch_size=batch_size,
            num_workers=num_workers,
            balance_tasks=balance_tasks,
        )


@dataclass
class DM05DebugTrainerConfig(_DM05TrainerConfig):
    """Keep the debug loop short and data loading in the debugged process."""

    num_train_steps: int = field(default=10)
    per_device_train_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=0)
    dataloader_persistent_workers: bool = field(default=False)
    dataloader_prefetch_factor: int | None = field(default=None)
    balance_tasks: bool = field(default=True)
    save_strategy: str = field(default="no")
    logging_steps: int = field(default=1)
    wandb_project: str | None = field(default=None)


@dataclass
class DM05DebugExp(_DM05Exp):
    trainer_config: DM05DebugTrainerConfig = field(
        default_factory=DM05DebugTrainerConfig
    )
    data_config: DM05DebugDataConfig = field(default_factory=DM05DebugDataConfig)


if __name__ == "__main__":
    exp = tyro.cli(DM05DebugExp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
