"""Training and serving entry point for the Piper representation ablation.

One file drives every rung; the rung is selected on the command line. Everything
that is not the variable under study -- horizon, learning rate, optimizer, steps,
seed, augmentation, state conditioning -- is fixed here so that differences
between runs are attributable to the representation.

Joint-space baseline (S0)::

    script/dm05_launcher.sh --exp playground/dm05_piper.py --task train \\
        --nproc_per_node 8 \\
        --data-config.dataset-name piper_fold_s0 \\
        --trainer-config.output-dir user_checkpoints/piper_s0

End-effector, base-frame delta -- the repo's existing convention (S1)::

        --data-config.dataset-name piper_fold_s1

End-effector, UMI body-frame delta with 6D rotation (S2)::

        --data-config.dataset-name piper_fold_s2 \\
        --data-config.relative-mode se3

Local-frame state plus inter-gripper pose (S2-pair), also supported for serving::

        --data-config.dataset-name piper_fold_s2_pair \\
        --data-config.relative-mode se3

This rung uses 23-D state (14 local EEF + 9 AUX) and 20-D body actions.

Unified frame plus inter-gripper pose (S3)::

        --data-config.dataset-name piper_fold_s3 \\
        --data-config.relative-mode se3

Gravity-aligned episode frame (S4), also supported for serving::

        --data-config.dataset-name piper_fold_s4 \\
        --data-config.relative-mode se3

The S4 robot client fixes W-to-G at rollout startup and handles both coordinate
conversions. This endpoint receives 14-D state in G and returns 20-D body actions.

Serving mirrors the training representation; see ``--inference-config`` defaults
and :meth:`DM05InferenceConfig._request_default_overrides`.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import tyro

from opendm.constants.robot import ActionMode
from opendm.dataset.piper_dual import (
    PIPER_EEF_PAIR_STATE_DESC,
    PIPER_EEF_STATE_DESC,
    PIPER_JOINT_STATE_DESC,
)
from opendm.exp.dm05_exp import (
    DM05DataConfig as _DM05DataConfig,
)
from opendm.exp.dm05_exp import (
    DM05Exp as _DM05Exp,
)
from opendm.exp.dm05_exp import (
    DM05InferenceConfig as _DM05InferenceConfig,
)
from opendm.exp.dm05_exp import (
    DM05ModelConfig as _DM05ModelConfig,
)
from opendm.exp.dm05_exp import (
    DM05OptimizerConfig as _DM05OptimizerConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)

# Number of state dims each rung produces, and the model-output width its action
# encoding needs. The relative SE(3) encoding emits 10 dims per arm
# (translation + 6D rotation + gripper); the vector encoding emits one delta per
# state dim.
RUNG_STATE_DESCS = {
    "piper_fold_s0": PIPER_JOINT_STATE_DESC,
    "piper_fold_s1": PIPER_EEF_STATE_DESC,
    "piper_fold_s2": PIPER_EEF_STATE_DESC,
    "piper_fold_s2_pair": PIPER_EEF_PAIR_STATE_DESC,
    "piper_fold_s3": PIPER_EEF_PAIR_STATE_DESC,
    "piper_fold_s3a": PIPER_EEF_STATE_DESC,
    "piper_fold_s4": PIPER_EEF_STATE_DESC,
    "piper_fold_s5": PIPER_JOINT_STATE_DESC,
}


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="piper_fold_s0")
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    # Kept on for every rung. Relative targets are defined against the current
    # state, so hiding the state from the prompt would change what each rung can
    # condition on -- and by different amounts per rung.
    add_state: bool = field(default=True)
    norm_stats_root: str = field(default="./norm_stats/piper")

    def _dataset_info(self):
        name = self.dataset_name
        if name not in RUNG_STATE_DESCS:
            raise ValueError(f"Unknown Piper ablation dataset: {name}")
        expected = (
            "se3"
            if name.removeprefix("piper_fold_") in ("s2", "s2_pair", "s3a", "s3", "s4")
            else "vector"
        )
        if self.relative_mode != expected:
            raise ValueError(f"{name} requires relative_mode={expected}")
        if self.action_mode != ActionMode.RELATIVE or not self.add_state:
            raise ValueError(
                "Piper ablations require relative actions and add_state=True"
            )
        return super()._dataset_info()


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    chunk_size: int = field(default=50)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=4e-5)
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=10000)
    num_train_steps: int = field(default=60000)
    save_only_model: bool = field(default=False)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    # 14 for the joint rungs and for the vector-delta EEF rung; 20 for the
    # relative-SE(3) rungs, whose action head predicts 10 dims per arm.
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )
    dataset_name: str = field(default="piper_fold_s0")

    def _initialize(self, **kwargs):
        # The robot host decodes the rung-specific deltas against its exact
        # observation anchor. Generic ActionAbsolute cannot decode SE(3).
        kwargs["use_absolute_action"] = False
        if self.compose_eef_rot:
            raise ValueError("Piper wire state already contains rotation vectors")
        super()._initialize(**kwargs)

    def _infer_legacy(self):
        from flask import jsonify

        return (
            jsonify({"error": "Use the Piper /v1/infer protocol and piper_rollout.py"}),
            400,
        )

    def _infer(self):
        from flask import jsonify, request
        from opendm.deploy.piper import contract

        spec = contract(self.dataset_name.removeprefix("piper_fold_"))
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body.get("piper") != spec:
            return (
                jsonify(
                    {
                        "error": "Piper client/server representation mismatch",
                        "piper": spec,
                    }
                ),
                400,
            )
        response = super()._infer()
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        payload["metadata"]["piper"] = spec
        return jsonify(payload)

    def _resolve_state_desc(self, robot_type):
        from opendm.constants.robot import RobotType

        if robot_type not in (None, RobotType.PIPER_DUAL.value):
            raise ValueError("Piper inference requires robot_type='Piper Dual'")
        # The generic registry describes Piper's 14 joint dimensions. Serving
        # an ablation must instead keep its EEF/AUX descriptor from the dataset.
        return list(RUNG_STATE_DESCS[self.dataset_name])

    def _prepare_input(self, body):
        from opendm.deploy.piper import contract, finite_array

        data = super()._prepare_input(body)
        spec = contract(self.dataset_name.removeprefix("piper_fold_"))
        finite_array(data["state"], (spec["state_dim"],))
        return data

    def _request_default_overrides(self) -> dict:
        from opendm.constants.robot import RobotType

        state_desc = RUNG_STATE_DESCS.get(self.dataset_name, PIPER_JOINT_STATE_DESC)
        control_mode = (
            "joint" if state_desc is PIPER_JOINT_STATE_DESC else "end effector"
        )
        return {
            "default_robot_type": RobotType.PIPER_DUAL.value,
            "default_state_desc": list(state_desc),
            "default_control_mode": control_mode,
        }


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def _initialize_inference_runtime(self):
        from opendm.deploy.piper import contract

        name = self.data_config.dataset_name
        spec = contract(name.removeprefix("piper_fold_"))
        expected = "se3" if spec["action_dim"] == 20 else "vector"
        if self.data_config.relative_mode != expected:
            raise ValueError(f"{name} requires --data-config.relative-mode {expected}")
        if self.data_config.action_mode != ActionMode.RELATIVE:
            raise ValueError("Piper deployment requires relative-action training")
        self.inference_config.dataset_name = name
        self.inference_config.output_action_dim = spec["action_dim"]
        super()._initialize_inference_runtime()


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
