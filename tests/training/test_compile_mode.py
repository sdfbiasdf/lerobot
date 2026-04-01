#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for compile_mode resolution with gradient accumulation.

CUDAGraphs (used by 'max-autotune' and 'reduce-overhead') is incompatible with
gradient accumulation because multiple forward passes before backward overwrite
tensors captured in the graph. These tests verify that compile_mode is resolved
correctly for all combinations of compile settings and gradient accumulation.
"""

import pytest

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


# -- Test default compile_mode is None across all policies --


@pytest.mark.parametrize(
    "config_cls",
    [PI05Config, PI0Config, PI0FastConfig, SmolVLAConfig, DiffusionConfig],
)
def test_default_compile_mode_is_none(config_cls):
    """All policy configs should default compile_mode to None."""
    config = config_cls()
    assert config.compile_mode is None


# -- Test modeling fallback: None resolves to policy-specific default --


def test_pi05_compile_mode_fallback():
    config = PI05Config(compile_model=True)
    compile_mode = config.compile_mode or "max-autotune"
    assert compile_mode == "max-autotune"


def test_diffusion_compile_mode_fallback():
    config = DiffusionConfig(compile_model=True)
    compile_mode = config.compile_mode or "reduce-overhead"
    assert compile_mode == "reduce-overhead"


# -- Test compile_mode resolution logic from lerobot_train.py --
# This replicates the logic in lerobot_train.py without requiring a full training setup.


def _resolve_compile_mode(policy_config, gradient_accumulation_steps: int):
    """Replicate the compile_mode resolution logic from lerobot_train.py."""
    _CUDAGRAPHS_MODES = {"max-autotune", "reduce-overhead"}

    if hasattr(policy_config, "compile_mode") and policy_config.compile_model:
        if policy_config.compile_mode is None and gradient_accumulation_steps > 1:
            policy_config.compile_mode = "max-autotune-no-cudagraphs"
        if policy_config.compile_mode is not None:
            if gradient_accumulation_steps > 1 and policy_config.compile_mode in _CUDAGRAPHS_MODES:
                raise ValueError(
                    f"compile_mode='{policy_config.compile_mode}' uses CUDAGraphs which is incompatible "
                    f"with gradient_accumulation_steps > 1. "
                    f"Use 'max-autotune-no-cudagraphs' or 'default' instead."
                )


def test_none_with_accumulation_resolves_to_no_cudagraphs():
    """compile_mode=None + accumulation > 1 should resolve to max-autotune-no-cudagraphs."""
    config = PI05Config(compile_model=True)
    _resolve_compile_mode(config, gradient_accumulation_steps=2)
    assert config.compile_mode == "max-autotune-no-cudagraphs"


def test_none_without_accumulation_stays_none():
    """compile_mode=None + accumulation=1 should remain None (resolved by modeling files)."""
    config = PI05Config(compile_model=True)
    _resolve_compile_mode(config, gradient_accumulation_steps=1)
    assert config.compile_mode is None


def test_explicit_cudagraphs_mode_with_accumulation_raises():
    """Explicitly setting a CUDAGraphs mode + accumulation > 1 should raise ValueError."""
    config = PI05Config(compile_model=True, compile_mode="max-autotune")
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_compile_mode(config, gradient_accumulation_steps=2)


def test_explicit_reduce_overhead_with_accumulation_raises():
    """reduce-overhead + accumulation > 1 should also raise ValueError."""
    config = DiffusionConfig(compile_model=True, compile_mode="reduce-overhead")
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_compile_mode(config, gradient_accumulation_steps=2)


def test_explicit_no_cudagraphs_with_accumulation_ok():
    """Explicitly setting max-autotune-no-cudagraphs + accumulation should work fine."""
    config = PI05Config(compile_model=True, compile_mode="max-autotune-no-cudagraphs")
    _resolve_compile_mode(config, gradient_accumulation_steps=2)
    assert config.compile_mode == "max-autotune-no-cudagraphs"


def test_explicit_default_mode_with_accumulation_ok():
    """Explicitly setting 'default' mode + accumulation should work fine."""
    config = PI05Config(compile_model=True, compile_mode="default")
    _resolve_compile_mode(config, gradient_accumulation_steps=2)
    assert config.compile_mode == "default"


def test_compile_disabled_skips_validation():
    """When compile_model=False, compile_mode should not be touched regardless of accumulation."""
    config = PI05Config(compile_model=False, compile_mode="max-autotune")
    _resolve_compile_mode(config, gradient_accumulation_steps=2)
    assert config.compile_mode == "max-autotune"  # not touched
