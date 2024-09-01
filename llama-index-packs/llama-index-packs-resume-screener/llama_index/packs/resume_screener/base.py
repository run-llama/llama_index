from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.bridge.pydantic import BaseModel, Field

# backwards compatibility
try:
    from llama_index.core.llms.llm import LLM
except ImportError:
    from llama_index.core.llms.base import LLM

QUERY_TEMPLATE = """
You are an expert resume reviewer.
You job is to decide if the candidate pass the resume screen given the job description and a list of criteria:

### Job Description
{job_description}

### Screening Criteria
{criteria_str}
"""


class CriteriaDecision(BaseModel):
    """The decision made based on a single criteria."""

    decision: bool = Field(description="The decision made based on the criteria")
    reasoning: str = Field(description="The reasoning behind the decision")


class ResumeScreenerDecision(BaseModel):
    """The decision made by the resume screener."""

    criteria_decisions: List[CriteriaDecision] = Field(
        description="The decisions made based on the criteria"
    )
    overall_reasoning: str = Field(
        description="The reasoning behind the overall decision"
    )
    overall_decision: bool = Field(
        description="The overall decision made based on the criteria"
    )


def _format_criteria_str(criteria: List[str]) -> str:
    criteria_str = ""
    for criterion in criteria:
        criteria_str += f"- {criterion}\n"
    return criteria_str


class ResumeScreenerPack(BaseLlamaPack):
    def __init__(
        self, job_description: str, criteria: List[str], llm: Optional[LLM] = None
    ) -> None:
        self.reader = PDFReader()
        llm = llm or OpenAI(model="gpt-4")
        Settings.llm = llm
        self.synthesizer = TreeSummarize(output_cls=ResumeScreenerDecision)
        criteria_str = _format_criteria_str(criteria)
        self.query = QUERY_TEMPLATE.format(
            job_description=job_description, criteria_str=criteria_str
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"reader": self.reader, "synthesizer": self.synthesizer}

    def run(self, resume_path: str, *args: Any, **kwargs: Any) -> Any:
        """Run pack."""
        docs = self.reader.load_data(Path(resume_path))
        output = self.synthesizer.synthesize(
            query=self.query,
            nodes=[NodeWithScore(node=doc, score=1.0) for doc in docs],
        )
        return output.response
