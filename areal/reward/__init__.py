from areal.utils import logging

logger = logging.getLogger("RewardUtils")

VALID_REWARD_FN = ["webgym"]


def get_custom_reward_fn(path: str, **kwargs):
    if "webgym" in path:
        from .webgym import webgym_reward_fn

        return webgym_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )
