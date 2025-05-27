"""
Mix Self Consistency Query Engine.

All prompts adapted from original paper by Liu et al. (2023):
https://arxiv.org/pdf/2312.16702v1.pdf
"""

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms import LLM
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.output_parsers.base import ChainableOutputParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.query_engine.custom import CustomQueryEngine
from llama_index.core.query_pipeline.query import QueryPipeline as QP
from pydantic import Field

# ===== Textual Reasoning =====

text_prompt_str = """\
You are an advanced AI capable of analyzing and understanding information within \
tables. Read the table below.

{table}

Based on the given table, answer the following question:

{question}

Let's think step by step, and then give the final answer. \
Ensure the final answer format is only \
"Final Answer: AnswerName1, AnswerName2..." form, no other form. \
And ensure the final answer is a number or entity names, as short as possible, \
without any explanation.
"""
text_prompt = PromptTemplate(template=text_prompt_str)


class FinalAnswerOutputParser(ChainableOutputParser):
    def parse(self, output: str) -> Optional[str]:
        lines = output.split("Final Answer:")
        if len(lines) > 1:
            return lines[-1].strip()
        return None

    def format(self, query: Any) -> str:
        return query


async def async_textual_reasoning(
    table: pd.DataFrame,
    query_str: str,
    llm: LLM,
    verbose: bool = False,
    temperature: float = 0.0,
) -> List[str]:
    # TODO: improve
    llm.temperature = temperature
    output_parser = FinalAnswerOutputParser()
    markdown_table = table.to_markdown()

    chain = QP(chain=[text_prompt, llm, output_parser], verbose=verbose)

    response = await chain.arun(
        question=query_str,
        table=markdown_table,
    )
    return str(response)


# ===== Symbolic Reasoning =====

# NOTE: this is adapted from the PyAgent prompt and pandas query engine prompt
pandas_prompt_str = """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.to_markdown())`:
{df_str}

Here's the input query: {query_str}.

Additional Guidelines:
- **Aggregated Rows**: Be cautious of rows that aggregate data such as 'total', 'sum', or 'average'.
Ensure these rows do not influence your results inappropriately.
- **Note**: All cells in the table should be considered as `object` data type, regardless of their
appearance.

Given the df information and the input query, please follow these instructions:
{instruction_str}

Output:
"""

pandas_prompt = PromptTemplate(template=pandas_prompt_str)


async def async_symbolic_reasoning(
    df: pd.DataFrame,
    query_str: str,
    llm: LLM,
    verbose: bool,
    temperature: float = 0.0,
) -> str:
    # TODO: improve
    llm.temperature = temperature

    query_engine = PandasQueryEngine(
        df=df,
        llm=llm,
        pandas_prompt=pandas_prompt,
        head=len(df),
        verbose=verbose,
    )
    response = await query_engine.aquery(query_str)
    return str(response)


# ===== Reasoning Aggregation =====
class AggregationMode(str, Enum):
    SELF_CONSISTENCY = "self-consistency"
    SELF_EVALUATION = "self-evaluation"
    NONE = "none"


self_evaluation_prompt_str = """\
Below is a markdown table:

{table}

You're tasked with answering the following question:

{question}

You have 2 answers derived by two different methods. Answer A was derived by prompting the AI to
think step-by-step. Answer B was derived by interacting with a Python Shell.

Answer A is {textual_answer}.
Answer B is {symbolic_answer}.

Your task is to determine which is the correct answer. It is crucial that you strictly adhere to the
following evaluation process:

1. **Preliminary Evaluation**: Begin by evaluating which of the two answers directly addresses the
question in a straightforward and unambiguous manner. A direct answer provides a clear response
that aligns closely with the query without introducing additional or extraneous details. If one
of the answers is not a direct response to the question, simply disregard it.
2. **Nature of the Question**: If both answers appear to be direct answers, then evaluate the nature
of the question. For tasks involving computation, counting, and column-locating, especially
when for extensive table, the Python Shell (Answer B) might be more precise. However, always
remain cautious if the Python Shell's output appears off (e.g., error messages, success
notifications, etc.). Such outputs may not be trustworthy for a correct answer.
3. **Final Verdict**: Finally, after thorough evaluation and explanation, provide your verdict
strictly following the given format:
- Use "[[A]]" if Answer A is correct.
- Use "[[B]]" if Answer B is correct.

Note:
1. Each method has its own strengths and weaknesses. Evaluate them with an unbiased perspective.
When in doubt, consider the nature of the question and lean towards the method that is most
suited for such queries.
2. Ensure that your verdict is provided after evaluation, at the end.
"""

self_evaluation_prompt = PromptTemplate(template=self_evaluation_prompt_str)


class EvalOutputParser(ChainableOutputParser):
    def parse(self, output: str) -> Optional[str]:
        if "[[A]]" in output:
            return "A"
        elif "[[B]]" in output:
            return "B"
        else:
            return None

    def format(self, query: Any) -> str:
        return query


def aggregate_self_evaluation(
    df: pd.DataFrame,
    query_str: str,
    text_result: str,
    symbolic_result: str,
    llm: LLM,
    verbose: bool = False,
) -> str:
    output_parser = EvalOutputParser()
    markdown_table = df.to_markdown()

    chain = QP(chain=[self_evaluation_prompt, llm, output_parser], verbose=verbose)
    response = chain.run(
        question=query_str,
        table=markdown_table,
        textual_answer=text_result,
        symbolic_answer=symbolic_result,
    )
    if str(response) == "A":
        return text_result
    elif str(response) == "B":
        return symbolic_result
    else:
        raise ValueError(f"Invalid response: {response}")


def aggregate_self_consistency(results: List[str]) -> str:
    # TODO: currently assumes exact match, can be improved with fuzzy matches
    counts = {}
    for result in results:
        if result in counts:
            counts[result] += 1
        else:
            counts[result] = 1
    return max(counts, key=counts.get)


def aggregate(
    table: pd.DataFrame,
    query_str: str,
    text_results: List[str],
    symbolic_results: List[str],
    llm: LLM,
    aggregation_mode: AggregationMode,
    verbose: bool = False,
) -> str:
    if verbose:
        print(f"Aggregation mode: {aggregation_mode}")
        print(f"Text results: {text_results}")
        print(f"Symbolic results: {symbolic_results}")

    if aggregation_mode == AggregationMode.SELF_EVALUATION:
        assert len(text_results) == 1 and len(symbolic_results) == 1, (
            "Must use exactly 1 text reasoning path and 1 symbolic reasoning path."
        )
        result = aggregate_self_evaluation(
            table,
            query_str,
            text_results[0],
            symbolic_results[0],
            llm,
            verbose=verbose,
        )
    elif aggregation_mode == AggregationMode.SELF_CONSISTENCY:
        result = aggregate_self_consistency(text_results + symbolic_results)
    elif aggregation_mode == AggregationMode.NONE:
        if len(symbolic_results) == 0 and len(text_results) == 1:
            result = text_results[0]
        elif len(text_results) == 0 and len(symbolic_results) == 1:
            result = symbolic_results[0]
        else:
            raise ValueError(
                "Must use exactly 1 text reasoning path or 1 symbolic reasoning path."
            )
    else:
        raise ValueError(f"Invalid aggregation mode: {aggregation_mode}")

    return result


class MixSelfConsistencyQueryEngine(CustomQueryEngine):
    df: pd.DataFrame = Field(..., description="Table (in pandas).")
    llm: LLM = Field(..., description="LLM to use.")
    verbose: bool = Field(
        default=False, description="Whether to print debug information."
    )
    text_paths: int = Field(default=5, description="Number of textual reasoning paths.")
    symbolic_paths: int = Field(
        default=5, description="Number of symbolic reasoning paths."
    )
    aggregation_mode: AggregationMode = Field(
        default=AggregationMode.SELF_CONSISTENCY,
        description="Aggregation mode.",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        llm: Optional[LLMType] = None,
        verbose: bool = False,
        normalize_table: bool = False,
        text_paths: int = 2,
        symbolic_paths: int = 2,
        aggregation_mode: AggregationMode = AggregationMode.SELF_CONSISTENCY,
        **kwargs: Any,
    ):
        llm = resolve_llm(llm)

        super().__init__(
            df=df,
            llm=llm,
            verbose=verbose,
            normalize_table=normalize_table,
            text_paths=text_paths,
            symbolic_paths=symbolic_paths,
            aggregation_mode=aggregation_mode,
            **kwargs,
        )

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        text_results: List[str] = []
        symbolic_results: List[str] = []

        if self.text_paths + self.symbolic_paths == 1:
            temperature = 0.0
        else:
            temperature = 0.8

        for ind in range(self.text_paths):
            if self.verbose:
                print(f"Textual Reasoning Path {ind + 1}/{self.text_paths}")
            response = asyncio.run(
                async_textual_reasoning(
                    self.df,
                    query_str,
                    self.llm,
                    self.verbose,
                    temperature=temperature,
                )
            )
            if self.verbose:
                print(f"Response: {response}")
                text_results.append(response)

        for ind in range(self.symbolic_paths):
            if self.verbose:
                print(f"Symbolic Reasoning Path {ind + 1}/{self.symbolic_paths}")
            response = asyncio.run(
                async_symbolic_reasoning(
                    self.df,
                    query_str,
                    self.llm,
                    self.verbose,
                    temperature=temperature,
                )
            )
            if self.verbose:
                print(f"Response: {response}")
            symbolic_results.append(response)

        return aggregate(
            self.df,
            query_str,
            text_results,
            symbolic_results,
            self.llm,
            self.aggregation_mode,
            verbose=self.verbose,
        )

    async def acustom_query(self, query_str: str) -> RESPONSE_TYPE:
        text_results: List[str] = []
        symbolic_results: List[str] = []
        tasks = []

        if self.text_paths + self.symbolic_paths == 1:
            temperature = 0.0
        else:
            temperature = 0.8

        for _ in range(self.text_paths):
            task = async_textual_reasoning(
                self.df,
                query_str,
                self.llm,
                self.verbose,
                temperature=temperature,
            )
            tasks.append(task)

        for _ in range(self.symbolic_paths):
            task = async_symbolic_reasoning(
                self.df, query_str, self.llm, self.verbose, temperature=temperature
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        for response in responses:
            if len(text_results) < self.text_paths:
                text_results.append(response)
            else:
                symbolic_results.append(response)

        return aggregate(
            self.df,
            query_str,
            text_results,
            symbolic_results,
            self.llm,
            self.aggregation_mode,
            verbose=self.verbose,
        )


class MixSelfConsistencyPack(BaseLlamaPack):
    """Mix Self Consistency Pack."""

    def __init__(
        self,
        table: pd.DataFrame,
        llm: Optional[LLMType] = None,
        verbose: bool = False,
        normalize_table: bool = False,
        text_paths: int = 2,
        symbolic_paths: int = 2,
        aggregation_mode: AggregationMode = AggregationMode.SELF_CONSISTENCY,
        **kwargs: Any,
    ) -> None:
        self.query_engine = MixSelfConsistencyQueryEngine(
            table=table,
            llm=llm,
            verbose=verbose,
            normalize_table=normalize_table,
            text_paths=text_paths,
            symbolic_paths=symbolic_paths,
            aggregation_mode=aggregation_mode,
            **kwargs,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "query_engine": self.query_engine,
            "llm": self.query_engine.llm,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        self.query_engine.query(*args, **kwargs)
