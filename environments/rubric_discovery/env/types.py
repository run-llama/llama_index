"""Shared type contracts for the rubric-discovery environment."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


class ExecutionBackend(str, enum.Enum):
    """Where candidate ``rubric_fn`` code is executed."""

    AUTO = "auto"
    SANDBOX = "sandbox"
    SUBPROCESS = "subprocess"


@dataclass(frozen=True)
class LabeledExample:
    """A single (input, response, score) training or test example.

    Attributes:
        input_text: The original input / prompt.
        response: The generated response to be scored.
        score: Ground-truth score in ``[0.0, 1.0]``.
        category: Optional category tag for filtering.
        metadata: Arbitrary extra metadata carried through the pipeline.
    """

    input_text: str
    response: str
    score: float
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        d: Dict[str, Any] = {
            "input_text": self.input_text,
            "response": self.response,
            "score": self.score,
        }
        if self.category:
            d["category"] = self.category
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabeledExample":
        """Deserialize from a dictionary."""
        return cls(
            input_text=data["input_text"],
            response=data["response"],
            score=float(data["score"]),
            category=data.get("category", ""),
            metadata=data.get("metadata", {}),
        )


# Type alias for a rubric function: (input_text, response) -> float
RubricFn = Callable[[str, str], float]


@dataclass
class EvaluationResult:
    """Result of evaluating a candidate rubric_fn on a set of examples.

    Attributes:
        predictions: Predicted scores for each example.
        labels: Ground-truth scores for each example.
        valid: Whether the rubric_fn executed without errors.
        error: Error message if execution failed.
    """

    predictions: List[float] = field(default_factory=list)
    labels: List[float] = field(default_factory=list)
    valid: bool = True
    error: Optional[str] = None


@dataclass
class DatasetRow:
    """A single row from the JSONL dataset, ready for RLMEnv consumption.

    Attributes:
        train_examples: Labeled examples shown to the model.
        test_examples: Held-out examples used for scoring.
        category: Category label for the row.
        metadata: Extra metadata (e.g. source environment info).
    """

    train_examples: List[LabeledExample]
    test_examples: List[LabeledExample]
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "train_examples": [e.to_dict() for e in self.train_examples],
            "test_examples": [e.to_dict() for e in self.test_examples],
            "category": self.category,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRow":
        """Deserialize from a dictionary."""
        return cls(
            train_examples=[
                LabeledExample.from_dict(e) for e in data["train_examples"]
            ],
            test_examples=[
                LabeledExample.from_dict(e) for e in data["test_examples"]
            ],
            category=data.get("category", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RubricDiscoveryConfig:
    """Configuration for the rubric-discovery environment.

    See the environment README for full descriptions of each parameter.
    """

    dataset_path: Optional[str] = None
    rlm_model: str = "gpt-4.1-mini"
    max_turns: int = 10
    categories: Optional[List[str]] = None
    max_examples: Optional[int] = None
    code_execution_timeout: int = 120
    max_sub_llm_parallelism: int = 5
    eval_backend: str = "auto"
    eval_tolerance: float = 0.3
    eval_timeout_s: int = 10
    validity_timeout_s: int = 5
    root_prompt_verbosity: str = "heavy"
    sub_prompt_verbosity: str = "medium"

    def get_execution_backend(self) -> ExecutionBackend:
        """Resolve the configured eval_backend string to an enum."""
        return ExecutionBackend(self.eval_backend)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RubricDiscoveryConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})
