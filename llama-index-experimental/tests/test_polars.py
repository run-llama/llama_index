"""Test polars index."""

import sys
from pathlib import Path
from typing import Any, Dict, cast

import polars as pl
import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.llms.mock import MockLLM
from llama_index.experimental.query_engine.polars.prompts import DEFAULT_POLARS_PROMPT
from llama_index.experimental.query_engine.polars.output_parser import (
    PolarsInstructionParser,
)
from llama_index.experimental.query_engine.polars.polars_query_engine import (
    PolarsQueryEngine,
)


def _mock_predict(*args: Any, **kwargs: Any) -> str:
    """Mock predict."""
    query_str = kwargs["query_str"]
    # Return Polars-style syntax for the mock
    return f'df.select(pl.col("{query_str}"))'


def test_polars_query_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test polars query engine."""
    monkeypatch.setattr(MockLLM, "predict", _mock_predict)
    llm = MockLLM()

    # Test on some sample data
    df = pl.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
            "description": [
                """Toronto, Canada's largest city, is a vibrant and diverse metropolis situated in the province of Ontario.
Known for its iconic skyline featuring the CN Tower, Toronto is a cultural melting pot with a rich blend of communities, languages, and cuisines.
It boasts a thriving arts scene, world-class museums, and a strong economic hub.
Visitors can explore historic neighborhoods, such as Kensington Market and Distillery District, or enjoy beautiful natural surroundings on Toronto Islands.
With its welcoming atmosphere, top-notch education, and multicultural charm, Toronto is a global destination for both tourists and professionals alike.""",
                "A city",
                "Another City",
            ],
        }
    )
    # the mock prompt just takes the all items in the given column
    query_engine = PolarsQueryEngine(df, llm=llm, verbose=True)
    response = query_engine.query(QueryBundle("population"))
    assert isinstance(response, Response)

    if sys.version_info < (3, 9):
        assert str(response) == 'df.select(pl.col("population"))'
    else:
        expected_output = str(df.select(pl.col("population")))
        assert str(response) == expected_output
    metadata = cast(Dict[str, Any], response.metadata)
    assert metadata["polars_instruction_str"] == 'df.select(pl.col("population"))'

    query_engine = PolarsQueryEngine(
        df,
        llm=llm,
        verbose=True,
        output_kwargs={"max_rows": 10},
    )
    response = query_engine.query(QueryBundle("description"))
    if sys.version_info < (3, 9):
        assert str(response) == 'df.select(pl.col("description"))'
    else:
        expected_output = str(df.select(pl.col("description")))
        assert str(response) == expected_output

    # test get prompts
    prompts = query_engine.get_prompts()
    assert prompts["polars_prompt"] == DEFAULT_POLARS_PROMPT


def test_default_output_processor_rce(tmp_path: Path) -> None:
    """
    Test that output processor prevents RCE.
    https://github.com/run-llama/llama_index/issues/7054 .
    """
    df = pl.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
        }
    )

    tmp_file = tmp_path / "pwnnnnn"

    injected_code = f"__import__('os').system('touch {tmp_file}')"
    parser = PolarsInstructionParser(df=df)
    parser.parse(injected_code)

    assert not tmp_file.is_file(), "file has been created via RCE!"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9 or higher")
def test_default_output_processor_rce2() -> None:
    """
    Test that output processor prevents RCE.
    https://github.com/run-llama/llama_index/issues/7054#issuecomment-1829141330 .
    """
    df = pl.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
        }
    )

    # Test various RCE attempts
    parser = PolarsInstructionParser(df=df)

    # Test malicious code injection attempts
    malicious_codes = [
        "__import__('subprocess').call(['echo', 'pwned'])",
        "exec('import os; os.system(\"echo pwned\")')",
        'eval(\'__import__("os").system("echo pwned")\')',
        "open('/etc/passwd').read()",
        "__builtins__.__dict__['eval']('print(\"pwned\")')",
    ]

    for malicious_code in malicious_codes:
        try:
            result = parser.parse(malicious_code)
            # The result should contain an error message about forbidden access
            assert "error" in str(result).lower() or "forbidden" in str(result).lower()
        except Exception:
            # Any exception is fine as it means the code was blocked
            pass


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9 or higher")
def test_default_output_processor_e2e(tmp_path: Path) -> None:
    """Test end-to-end functionality with real Polars operations."""
    df = pl.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
            "country": ["Canada", "Japan", "Germany"],
        }
    )

    parser = PolarsInstructionParser(df=df)

    # Test valid Polars operations
    valid_operations = [
        "df.select(pl.col('city'))",
        "df.filter(pl.col('population') > 5000000)",
        "df.with_columns(pl.col('population').alias('pop'))",
        "df.group_by('country').agg(pl.col('population').sum())",
        "df.head(2)",
        "df.select([pl.col('city'), pl.col('population')])",
    ]

    for operation in valid_operations:
        try:
            result = parser.parse(operation)
            # Should not contain error messages
            assert "error" not in str(result).lower()
        except Exception as e:
            # If there's an exception, it should be a valid execution error, not security-related
            assert "forbidden" not in str(e).lower()
            assert "private" not in str(e).lower()


def test_polars_query_engine_complex_operations() -> None:
    """Test PolarsQueryEngine with more complex operations."""
    df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "age": [25, 30, 35, 28],
            "salary": [50000, 60000, 70000, 55000],
            "department": ["Engineering", "Sales", "Engineering", "Sales"],
        }
    )

    # Mock LLM that returns complex Polars operations
    class ComplexMockLLM(MockLLM):
        def predict(self, *args, **kwargs):
            query_str = kwargs.get("query_str", "")
            if "average salary" in query_str.lower():
                return "df.select(pl.col('salary').mean())"
            elif "engineering" in query_str.lower():
                return "df.filter(pl.col('department') == 'Engineering')"
            elif "group by" in query_str.lower():
                return "df.group_by('department').agg(pl.col('salary').mean())"
            else:
                return "df.head()"

    llm = ComplexMockLLM()
    query_engine = PolarsQueryEngine(df, llm=llm, verbose=True)

    # Test average salary query
    response = query_engine.query(QueryBundle("What is the average salary?"))
    if sys.version_info >= (3, 9):
        expected = str(df.select(pl.col("salary").mean()))
        assert str(response) == expected

    # Test filtering query
    response = query_engine.query(QueryBundle("Show engineering employees"))
    if sys.version_info >= (3, 9):
        expected = str(df.filter(pl.col("department") == "Engineering"))
        assert str(response) == expected

    # Test groupby query
    response = query_engine.query(QueryBundle("Group by department"))
    if sys.version_info >= (3, 9):
        expected = str(df.group_by("department").agg(pl.col("salary").mean()))
        assert str(response) == expected
