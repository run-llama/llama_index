"""Data-frame program.

NOTE: inspired from jxnl's repo example here:
https://github.com/jxnl/openai_function_call/blob/main/auto_dataframe.py


"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class DataFrameRow(BaseModel):
    """Row in a DataFrame.

    Consists of a `row_values` field which is a list of dictionaries,.

    """

    row_values: List[Dict] = Field(
        ...,
        description="""List of dictionaries, where each dictionary is a \
        column-value pair.""",
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

    rows: List[Dict] = Field(
        ...,
        description="""List of rows as dictionaries, where each dictionary \
            contains key-value pairs of column names/values.""",
    )
    columns: List[DataFrameColumn] = Field(..., description="List of column names.")


class DataFrameTestRow(BaseModel):
    """Row in a DataFrame."""

    row_values: List[str] = Field(
        ...,
        description="List of row values, where each value corresponds to a row key." "",
    )


class DataFrameWithColumns(BaseModel):
    """Data-frame with rows. Assume column names are already known beforehand."""

    # rows: List[Dict] = Field(
    #     ...,
    #     description="""List of rows as dictionaries, where each dictionary \
    #         contains key-value pairs of column names/values.""",
    # )
    rows: List[DataFrameTestRow] = Field(..., description="""List of row objects.""")


# class DataFrameProgram(BasePydanticProgram):
#     """A program that returns a data-frame."""

#     def __init__(self, base_program: BasePydanticProgram[BaseModel]) -> None:
#         """Initialize params."""
#         self._base_program = base_program

#     @property
#     def output_cls(self) -> Type[DataFrame]:
#         return DataFrame

#     def __call__(self, *args: Any, **kwds: Any) -> DataFrame:

#         return DataFrame(*args, **kwds)
