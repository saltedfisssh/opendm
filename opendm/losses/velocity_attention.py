"""Velocity-field action attention (AttenA+) for PyTorch models."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

WeightStrategy = Literal["inverse", "inverse_squared", "exp_decay", "log"]
LossType = Literal["l1", "mse"]
Reduction = Literal["mean", "none"]


class VelocityAttention:
    """Assign larger loss weights to low-speed action timesteps.

    Weights are computed from the leading ``joint_dims`` dimensions of the
    normalized ground-truth action and broadcast across all action dimensions.
    This follows the canonical AttenA+ implementation while accepting OpenDM's
    per-dimension ``action_mask``.
    """

    _WEIGHT_STRATEGIES = {"inverse", "inverse_squared", "exp_decay", "log"}

    def __init__(
        self,
        weight_strategy: WeightStrategy = "inverse_squared",
        clip_max_weight: float = 2.0,
        epsilon: float = 1e-3,
        alpha: float = 5.0,
        normalize_weights: bool = True,
        joint_dims: int = 6,
        joint_indices: Sequence[int] | None = None,
    ) -> None:
        if weight_strategy not in self._WEIGHT_STRATEGIES:
            raise ValueError(f"Unknown weight_strategy: {weight_strategy!r}")
        if clip_max_weight < 1.0:
            raise ValueError("clip_max_weight must be >= 1.0")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if joint_dims <= 0:
            raise ValueError("joint_dims must be > 0")
        if joint_indices is not None:
            joint_indices = tuple(joint_indices)
            if not joint_indices:
                raise ValueError("joint_indices must not be empty")
            if any(index < 0 for index in joint_indices):
                raise ValueError("joint_indices must contain non-negative indices")
            if len(set(joint_indices)) != len(joint_indices):
                raise ValueError("joint_indices must not contain duplicates")

        self.weight_strategy = weight_strategy
        self.clip_max_weight = clip_max_weight
        self.epsilon = epsilon
        self.alpha = alpha
        self.normalize_weights = normalize_weights
        self.joint_dims = joint_dims
        self.joint_indices = joint_indices

    def compute_weights(
        self,
        ground_truth_actions: Tensor,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Return weights shaped ``(B, T, 1)`` for actions shaped ``(B, T, D)``."""
        if ground_truth_actions.ndim != 3:
            raise ValueError("ground_truth_actions must have shape (B, T, D)")
        action_dim = ground_truth_actions.shape[-1]
        if self.joint_indices is None and self.joint_dims > action_dim:
            raise ValueError(
                f"joint_dims={self.joint_dims} exceeds action dimension "
                f"{action_dim}"
            )
        if self.joint_indices is not None and max(self.joint_indices) >= action_dim:
            raise ValueError(
                f"joint_indices contains {max(self.joint_indices)}, but action "
                f"dimension is {action_dim}"
            )

        if self.joint_indices is None:
            joint_actions = ground_truth_actions[..., : self.joint_dims]
            joint_mask = (
                None
                if action_mask is None
                else action_mask[..., : self.joint_dims]
            )
        else:
            index = torch.tensor(
                self.joint_indices,
                device=ground_truth_actions.device,
                dtype=torch.long,
            )
            joint_actions = ground_truth_actions.index_select(-1, index)
            joint_mask = (
                None if action_mask is None else action_mask.index_select(-1, index)
            )
        if action_mask is not None:
            if action_mask.shape != ground_truth_actions.shape:
                raise ValueError("action_mask must have the same shape as actions")
            joint_actions = joint_actions * joint_mask.to(
                dtype=joint_actions.dtype
            )

        speed = torch.linalg.vector_norm(joint_actions, dim=-1, keepdim=True)
        speed = speed.clamp_min(self.epsilon)
        weights = self._speed_to_weight(speed).clamp(
            min=1.0 / self.clip_max_weight,
            max=self.clip_max_weight,
        )
        if self.normalize_weights:
            weights = weights / self.clip_max_weight * 2.0
        return weights

    def weighted_loss(
        self,
        ground_truth: Tensor,
        predicted: Tensor,
        target: Tensor | None = None,
        loss_type: LossType = "mse",
        action_mask: Tensor | None = None,
        reduction: Reduction = "mean",
    ) -> Tensor:
        """Compute an AttenA+-weighted action loss.

        ``ground_truth`` determines weights. ``target`` is the actual regression
        target and defaults to ``ground_truth``. For flow matching, pass the
        noise-minus-action velocity field as ``target``.
        """
        loss_target = ground_truth if target is None else target
        if predicted.shape != loss_target.shape or predicted.shape != ground_truth.shape:
            raise ValueError("ground_truth, predicted, and target must share a shape")
        if loss_type == "l1":
            errors = (predicted - loss_target).abs()
        elif loss_type == "mse":
            errors = (predicted - loss_target).square()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

        weights = self.compute_weights(ground_truth, action_mask=action_mask)
        weighted_errors = errors * weights
        if action_mask is not None:
            mask = action_mask.to(device=errors.device, dtype=errors.dtype)
            numerator = (weighted_errors * mask).sum(dim=(1, 2))
            denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
            per_sample = numerator / denominator
        else:
            per_sample = weighted_errors.mean(dim=(1, 2))

        if reduction == "mean":
            return per_sample.mean()
        if reduction == "none":
            return per_sample
        raise ValueError(f"Unknown reduction: {reduction!r}")

    def _speed_to_weight(self, speed: Tensor) -> Tensor:
        if self.weight_strategy == "inverse":
            return speed.reciprocal()
        if self.weight_strategy == "inverse_squared":
            return speed.square().reciprocal()
        if self.weight_strategy == "exp_decay":
            return torch.exp(-self.alpha * speed)
        return torch.log1p(speed).reciprocal()
