"""Dataset IO: loading and saving JSONL datasets for rubric discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from environments.rubric_discovery.env.types import DatasetRow, LabeledExample

# Default dataset path (relative to this file's package)
_DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "rubric_discovery_dataset.jsonl",
)


def load_dataset(
    path: Optional[str] = None,
    categories: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
) -> List[DatasetRow]:
    """Load a JSONL dataset of rubric-discovery rows.

    Args:
        path: Path to the JSONL file. Defaults to the built-in dataset.
        categories: Optional category filter; only rows matching these
                    categories are returned.
        max_examples: Optional cap on the number of rows returned.

    Returns:
        A list of ``DatasetRow`` instances.
    """
    dataset_path = path or _DEFAULT_DATASET
    rows: List[DatasetRow] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            row = DatasetRow.from_dict(data)

            # Category filter
            if categories and row.category not in categories:
                continue

            rows.append(row)

            # Early exit if capped
            if max_examples is not None and len(rows) >= max_examples:
                break

    return rows


def save_dataset(rows: List[DatasetRow], path: str) -> None:
    """Save dataset rows to a JSONL file.

    Args:
        rows: List of ``DatasetRow`` instances to write.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def get_default_dataset_path() -> str:
    """Return the path to the built-in default dataset."""
    return _DEFAULT_DATASET
