# DM05 AttenA+ Velocity Attention

OpenDM integrates AttenA+ as an optional velocity-weighted flow-matching loss. Weights are computed from the leading joint dimensions of the normalized ground-truth action, while the MSE target remains `noise - action`. Model architecture and inference are unchanged, and the feature defaults to disabled for checkpoint compatibility.

Enable the paper-default setup with:

```bash
script/dm05_launcher.sh \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.use-velocity-attention true \
  --model-config.velocity-weight-strategy inverse_squared \
  --model-config.velocity-clip-max-weight 2.0 \
  --model-config.velocity-joint-dims 6
```

Available model options are `use_velocity_attention`, `velocity_weight_strategy` (`inverse`, `inverse_squared`, `exp_decay`, or `log`), `velocity_clip_max_weight`, `velocity_epsilon`, `velocity_alpha`, `velocity_normalize_weights`, and `velocity_joint_dims`. These values are persisted in the model checkpoint.

During training, OpenDM derives exact joint indices from the dataset
`state_desc`, excluding grippers. Table30 v2 Aloha and dos_w1 use
`[left 6 joints, left gripper, right 6 joints, right gripper]`, which resolves
to `[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]`.

The `velocity_joint_indices` option can explicitly override this selection.
When no indices or dataset semantics are available, `velocity_joint_dims`
remains the backward-compatible leading-dimension fallback.

The reusable API is:

```python
from opendm.losses import VelocityAttention

loss = VelocityAttention(
    joint_indices=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
).weighted_loss(
    ground_truth=actions,
    predicted=v_t,
    target=noise - actions,
    loss_type="mse",
    action_mask=action_mask,
)
```

Run the focused tests with:

```bash
conda run -n opendm python -m pytest tests/test_velocity_attention.py -q
```
