from typing import TYPE_CHECKING, Optional

from areal.api.cli_args import _DatasetConfig
from areal.utils import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

VALID_DATASETS = ["webgym"]

logger = logging.getLogger("Dataset")


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if "webgym" in path and type == "rl":
        from .webgym import get_webgym_rl_dataset

        return get_webgym_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Dataset {path} with split {split} and training type {type} is not supported. "
            f"Supported datasets are: {VALID_DATASETS}. "
        )


def get_custom_dataset(
    split: str | None = None,
    dataset_config: _DatasetConfig | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if dataset_config is not None:
        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

    logger.warning("dataset_config is not provided")
    return _get_custom_dataset(
        split=split,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )
