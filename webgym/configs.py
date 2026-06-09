from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig
from areal.utils.test_rollout import TestRolloutConfig


@dataclass
class WebGymConfig(GRPOConfig):
    """Extends GRPOConfig with WebGym-specific settings."""

    webgym: dict[str, Any] = field(default_factory=dict)
    openai_config: dict[str, Any] | None = field(default=None)
    test_rollout: TestRolloutConfig = field(default_factory=TestRolloutConfig)
