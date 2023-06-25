"""Data-frame output parser.

NOTE: inspired from jxnl's repo example here:
https://github.com/jxnl/openai_function_call/blob/main/auto_dataframe.py


"""

from typing import Optional, List, Any, Type
from pydantic import BaseModel, Field
from llama_index.types import BaseOutputParser
from llama_index.output_parsers.pydantic_program import PydanticProgramOutputParser
from llama_index.program.base_program import BasePydanticProgram
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.prompts.prompts import Prompt
import pandas as pd


class DataFrameRow(BaseModel):
    """Row in a DataFrame."""

    row_values: List[Any] = Field(
        ...,
        description="List of row values, where each value corresponds to a row key.",
    )


class DataFrameColumn(BaseModel):
    """Column in a DataFrame."""

    column_name: str = Field(..., description="Column name.")
    column_desc: Optional[str] = Field(..., description="Column description.")


class DataFrame(BaseModel):
    """Data-frame class.

    Consists of a `rows` field which is a list of dictionaries,
    as well as a `columns` field which is a list of column names.

    """

    description: Optional[str] = None

    columns: List[DataFrameColumn] = Field(..., description="List of column names.")
    rows: List[DataFrameRow] = Field(
        ...,
        description="""List of DataFrameRow objects. Each DataFrameRow contains \
        valuesin order of the data frame column.""",
    )

    def to_df(self) -> pd.DataFrame:
        """To dataframe."""
        return pd.DataFrame(
            [row.row_values for row in self.rows],
            columns=[col.column_name for col in self.columns],
        )


class DataFrameRowsOnly(BaseModel):
    """Data-frame with rows. Assume column names are already known beforehand."""

    rows: List[DataFrameRow] = Field(..., description="""List of row objects.""")

    def to_df(self, existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """To dataframe."""
        if existing_df is None:
            return pd.DataFrame([row.row_values for row in self.rows])
        else:
            new_df = pd.DataFrame([row.row_values for row in self.rows])
            new_df.columns = existing_df.columns
            # assume row values are in order of column names
            return existing_df.append(new_df, ignore_index=True)


DEFAULT_FULL_DF_PARSER_TMPL = """
Please extract the following query into a structured data.
Query: {input_str}.
Please extract both the set of column names and row names.
"""

DEFAULT_ROWS_DF_PARSER_TMPL = """
Please extract the following query into structured data.
Query: {input_str}.
The column schema is the following: {column_schema}.
"""


class DFFullOutputParser(BaseOutputParser):
    """Full DF output parser.

    Extracts text into a schema.

    """

    def __init__(
        self,
        pydantic_program_cls: Type[BasePydanticProgram],
        df_parser_template_str: str = DEFAULT_FULL_DF_PARSER_TMPL,
        input_key: str = "input_str",
    ) -> None:
        """Init params."""
        pydantic_program = pydantic_program_cls.from_defaults(
            DataFrame, df_parser_template_str
        )
        self._validate_program(pydantic_program)
        self._pydantic_parser = PydanticProgramOutputParser(pydantic_program, input_key)

    def _validate_program(self, pydantic_program: BasePydanticProgram) -> None:
        if pydantic_program.output_cls != DataFrame:
            raise ValueError("Output class of pydantic program must be `DataFrame`.")

    @classmethod
    def from_defaults(
        cls,
        pydantic_program_cls: Optional[Type[BasePydanticProgram]] = None,
        df_parser_template_str: str = DEFAULT_FULL_DF_PARSER_TMPL,
        input_key: str = "input_str",
    ) -> "DFFullOutputParser":
        """Full DF output parser."""
        pydantic_program_cls = pydantic_program_cls or OpenAIPydanticProgram

        return cls(
            pydantic_program_cls,
            df_parser_template_str=df_parser_template_str,
            input_key=input_key,
        )

    def parse(self, output: str) -> DataFrame:
        """Parse, validate, and correct errors programmatically."""
        return self._pydantic_parser.parse(output)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError("`format` method not supported for parser.")


class DFRowsOutputParser(BaseOutputParser):
    """DF Rows output parser.

    Given DF schema, extract text into a set of rows.

    """

    def __init__(
        self,
        pydantic_program_cls: Type[BasePydanticProgram],
        df_parser_template_str: str = DEFAULT_ROWS_DF_PARSER_TMPL,
        column_schema: Optional[str] = None,
        input_key: str = "input_str",
    ) -> None:
        """Init params."""
        # partial format df parser template string
        # NOTE: hack where we use prompt class to partial format
        orig_prompt = Prompt(df_parser_template_str)
        new_prompt = Prompt.from_prompt(
            orig_prompt.partial_format(
                column_schema=column_schema,
            )
        )

        pydantic_program = pydantic_program_cls.from_defaults(
            DataFrameRowsOnly, new_prompt.original_template
        )
        self._validate_program(pydantic_program)

        self._pydantic_parser = PydanticProgramOutputParser(pydantic_program, input_key)

    def _validate_program(self, pydantic_program: BasePydanticProgram) -> None:
        if pydantic_program.output_cls != DataFrameRowsOnly:
            raise ValueError(
                "Output class of pydantic program must be `DataFramRowsOnly`."
            )

    @classmethod
    def from_defaults(
        cls,
        pydantic_program_cls: Optional[Type[BasePydanticProgram]] = None,
        df_parser_template_str: str = DEFAULT_ROWS_DF_PARSER_TMPL,
        df: Optional[pd.DataFrame] = None,
        column_schema: Optional[str] = None,
        input_key: str = "input_str",
        **kwargs: Any
    ) -> "DFRowsOutputParser":
        """Rows DF output parser."""
        pydantic_program_cls = pydantic_program_cls or OpenAIPydanticProgram

        # either one of df or column_schema needs to be specified
        if df is None and column_schema is None:
            raise ValueError(
                "Either `df` or `column_schema` must be specified for "
                "DFRowsOutputParser."
            )
        # first, inject the column schema into the template string
        if column_schema is None:
            assert df is not None
            # by default, show column schema and some example values
            column_schema = ", ".join(df.columns)

        return cls(
            pydantic_program_cls,
            df_parser_template_str=df_parser_template_str,
            column_schema=column_schema,
            input_key=input_key,
        )

    def parse(self, output: str) -> BasePydanticProgram:
        """Parse, validate, and correct errors programmatically."""
        return self._pydantic_parser.parse(output)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError("`format` method not supported for parser.")
