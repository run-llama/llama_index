from typing import Any, List, Optional, Type, cast

import pandas as pd

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.types import BasePydanticProgram


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


class DataFrameValuesPerColumn(BaseModel):
    """Data-frame as a list of column objects.

    Each column object contains a list of values. Note that they can be
    of variable length, and so may not be able to be converted to a dataframe.

    """

    columns: List[DataFrameRow] = Field(..., description="""List of column objects.""")


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


class DFFullProgram(BasePydanticProgram[DataFrame]):
    """Data-frame program.

    Extracts text into a schema + datapoints.

    """

    def __init__(
        self,
        pydantic_program_cls: Type[BaseLLMFunctionProgram],
        df_parser_template_str: str = DEFAULT_FULL_DF_PARSER_TMPL,
        input_key: str = "input_str",
        **program_kwargs: Any,
    ) -> None:
        """Init params."""
        pydantic_program = pydantic_program_cls.from_defaults(
            DataFrame, df_parser_template_str, **program_kwargs
        )
        self._validate_program(pydantic_program)
        self._pydantic_program = pydantic_program
        self._input_key = input_key

    @classmethod
    def from_defaults(
        cls,
        pydantic_program_cls: Optional[Type[BaseLLMFunctionProgram]] = None,
        df_parser_template_str: str = DEFAULT_FULL_DF_PARSER_TMPL,
        input_key: str = "input_str",
    ) -> "DFFullProgram":
        """Full DF output parser."""
        pydantic_program_cls = pydantic_program_cls or OpenAIPydanticProgram

        return cls(
            pydantic_program_cls,
            df_parser_template_str=df_parser_template_str,
            input_key=input_key,
        )

    def _validate_program(self, pydantic_program: BasePydanticProgram) -> None:
        if pydantic_program.output_cls != DataFrame:
            raise ValueError("Output class of pydantic program must be `DataFrame`.")

    @property
    def output_cls(self) -> Type[DataFrame]:
        """Output class."""
        return DataFrame

    def __call__(self, *args: Any, **kwds: Any) -> DataFrame:
        """Call."""
        if self._input_key not in kwds:
            raise ValueError(f"Input key {self._input_key} not found in kwds.")
        result = self._pydantic_program(**{self._input_key: kwds[self._input_key]})
        return cast(DataFrame, result)


class DFRowsProgram(BasePydanticProgram[DataFrameRowsOnly]):
    """DF Rows output parser.

    Given DF schema, extract text into a set of rows.

    """

    def __init__(
        self,
        pydantic_program_cls: Type[BaseLLMFunctionProgram],
        df_parser_template_str: str = DEFAULT_ROWS_DF_PARSER_TMPL,
        column_schema: Optional[str] = None,
        input_key: str = "input_str",
        **program_kwargs: Any,
    ) -> None:
        """Init params."""
        # partial format df parser template string with column schema
        prompt_template_str = df_parser_template_str.replace(
            "{column_schema}", column_schema or ""
        )

        pydantic_program = pydantic_program_cls.from_defaults(
            DataFrameRowsOnly, prompt_template_str, **program_kwargs
        )
        self._validate_program(pydantic_program)
        self._pydantic_program = pydantic_program
        self._input_key = input_key

    def _validate_program(self, pydantic_program: BasePydanticProgram) -> None:
        if pydantic_program.output_cls != DataFrameRowsOnly:
            raise ValueError(
                "Output class of pydantic program must be `DataFramRowsOnly`."
            )

    @classmethod
    def from_defaults(
        cls,
        pydantic_program_cls: Optional[Type[BaseLLMFunctionProgram]] = None,
        df_parser_template_str: str = DEFAULT_ROWS_DF_PARSER_TMPL,
        df: Optional[pd.DataFrame] = None,
        column_schema: Optional[str] = None,
        input_key: str = "input_str",
        **kwargs: Any,
    ) -> "DFRowsProgram":
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
            **kwargs,
        )

    @property
    def output_cls(self) -> Type[DataFrameRowsOnly]:
        """Output class."""
        return DataFrameRowsOnly

    def __call__(self, *args: Any, **kwds: Any) -> DataFrameRowsOnly:
        """Call."""
        if self._input_key not in kwds:
            raise ValueError(f"Input key {self._input_key} not found in kwds.")
        result = self._pydantic_program(**{self._input_key: kwds[self._input_key]})
        return cast(DataFrameRowsOnly, result)
