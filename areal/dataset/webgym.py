"""WebGym dataset loader for AReaL RL training.

Loads task JSONL files used by the WebGym web automation environment.
Each sample provides task metadata; the actual multi-turn conversation
is built dynamically in the workflow (since screenshots change each step).
"""

import json
import os

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("WebGymDataset")


def get_webgym_rl_dataset(
    path: str,
    split: str | None = None,
    processor=None,
    max_length: int | None = None,
    **kwargs,
) -> Dataset:
    """Load WebGym task JSONL file as a HuggingFace Dataset.

    Parameters
    ----------
    path : str
        Path to the JSONL task file (e.g., ``train_tasks.jsonl``).
    split : str, optional
        Unused for file-based datasets; kept for API compatibility.
    processor : optional
        Unused; kept for API compatibility with vision dataset loaders.
    max_length : int, optional
        Unused; kept for API compatibility.

    Returns
    -------
    Dataset
        HuggingFace Dataset with fields: task_name, website, domain,
        subdomain, difficulty, evaluator_reference, reference_answer,
        task_id, messages.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"WebGym task file not found: {path}. "
            "Please provide a valid path to a JSONL task file."
        )

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(row)

    logger.info(f"Loaded {len(records)} tasks from {path}")

    def process(row):
        task_name = row.get("task_name", row.get("task", ""))
        website = row.get("website", "")
        domain = row.get("domain", "")
        subdomain = row.get("subdomain", "")
        difficulty = row.get("difficulty", "")
        evaluator_reference = row.get("evaluator_reference", [])
        reference_answer = row.get("reference_answer", None)
        task_id = row.get("task_id", None)

        # Build initial messages for AReaL data pipeline compatibility.
        # The actual multi-turn conversation is constructed dynamically
        # in the workflow using ContextManager + live screenshots.
        messages = [
            {
                "role": "user",
                "content": f"Complete the following web task: {task_name}",
            }
        ]

        return {
            "task_name": task_name,
            "website": website,
            "domain": domain,
            "subdomain": subdomain,
            "difficulty": difficulty,
            "evaluator_reference": json.dumps(evaluator_reference)
            if isinstance(evaluator_reference, (list, dict))
            else str(evaluator_reference),
            "reference_answer": reference_answer or "",
            "task_id": str(task_id) if task_id is not None else "",
            "messages": messages,
        }

    processed = [process(r) for r in records]
    dataset = Dataset.from_list(processed)

    logger.info(
        f"WebGym RL dataset ready: {len(dataset)} tasks "
        f"(split={split}, path={path})"
    )
    return dataset
