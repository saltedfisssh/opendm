"""Keep the CUDA Graph diffusion step numerically aligned with eager decoding."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from opendm.model.dm05.dm05_arch import (
    DM05ForConditionalGeneration,
    DM05SuffixGraphProfile,
)


@pytest.mark.parametrize("policy", ["bf16_mixed", "fp32_mixed"])
@pytest.mark.parametrize("masked", [False, True])
def test_suffix_graph_steps_match_eager_precision_and_time_conditioning(policy, masked):
    # Exercise the real eager loop, projections, time MLP and graph step with a
    # small deterministic expert, without allocating the full VLM or requiring CUDA.
    torch.manual_seed(31)
    dtype = torch.bfloat16 if policy == "bf16_mixed" else torch.float32

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8).to(dtype)

        def forward(self, *, suffix_embeds, adarms_cond, **kwargs):
            return torch.tanh(self.proj(suffix_embeds) + adarms_cond[:, None])

    model = SimpleNamespace(
        precision_policy=policy,
        model=SimpleNamespace(
            config=SimpleNamespace(chunk_size=3, action_dim=4),
            action_in_proj=nn.Linear(4, 8).to(dtype),
            action_out_proj=nn.Linear(8, 4).to(dtype),
            time_mlp_in=nn.Linear(8, 8).to(dtype),
            time_mlp_out=nn.Linear(8, 8).to(dtype),
            action_expert=Expert(),
            vlm=SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(padding_idx=0))
            ),
        ),
        _can_use_suffix_graph=lambda value: False,
        _compute_prefix_cache=lambda **kwargs: (None, torch.zeros(1, 5, 8)),
        _extract_prefix_cache_tensors=lambda cache: ((), ()),
    )
    for name in (
        "_inference_action_impl",
        "_action_input_proj",
        "_action_output_proj",
        "_suffix_hidden_dtype",
        "_build_adarms_cond",
        "_build_suffix_position_ids",
        "_suffix_forward",
        "_run_suffix_graph_step",
    ):
        setattr(
            model, name, MethodType(getattr(DM05ForConditionalGeneration, name), model)
        )
    action_mask = torch.tensor([[[1, 1, 1, 0]]], dtype=dtype) if masked else None
    with torch.no_grad():
        torch.manual_seed(7)
        expected = model._inference_action_impl(
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            diffusion_steps=10,
            action_mask=action_mask,
        )
        torch.manual_seed(7)
        initial_noise = torch.randn(1, 3, 4, dtype=dtype)
        profile = DM05SuffixGraphProfile(
            prefix_len=5,
            diffusion_steps=10,
            prefix_cache_keys=(),
            prefix_cache_values=(),
            attention_mask=None,
            position_ids=None,
            state=initial_noise.float(),
            time=torch.ones(1, dtype=dtype),
            action_mask=action_mask,
        )
        time_value = 1.0
        for _ in range(10):
            profile.time.fill_(time_value)
            model._run_suffix_graph_step(profile)
            time_value -= 0.1
    assert profile.state.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(profile.state, expected, atol=0, rtol=0)
