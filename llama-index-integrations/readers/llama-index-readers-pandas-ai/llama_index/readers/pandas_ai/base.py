"""Pandas AI loader."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import PandasCSVReader


class PandasAIReader(BaseReader):
    r"""Pandas AI reader.

    Light wrapper around https://github.com/gventuri/pandas-ai.

    Args:
        llm (Optional[pandas.llm]): LLM to use. Defaults to None.
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        col_joiner (str): Separator to use for joining cols per row.
            Set to ", " by default.

        row_joiner (str): Separator to use for joining each row.
            Only used when `concat_rows=True`.
            Set to "\n" by default.

        pandas_config (dict): Options for the `pandas.read_csv` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
            for more information.
            Set to empty dict by default, this means pandas will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        concat_rows: bool = True,
        col_joiner: str = ", ",
        row_joiner: str = "\n",
        pandas_config: dict = {},
    ) -> None:
        """Init params."""
        try:
            from pandasai import PandasAI
            from pandasai.llm.openai import OpenAI
        except ImportError:
            raise ImportError("Please install pandasai to use this reader.")

        self._llm = llm or OpenAI()
        self._pandas_ai = PandasAI(llm)

        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def run_pandas_ai(
        self,
        initial_df: pd.DataFrame,
        query: str,
        is_conversational_answer: bool = False,
    ) -> Any:
        """Load dataframe."""
        return self._pandas_ai.run(
            initial_df, prompt=query, is_conversational_answer=is_conversational_answer
        )

    def load_data(
        self,
        initial_df: pd.DataFrame,
        query: str,
        is_conversational_answer: bool = False,
    ) -> List[Document]:
        """Parse file."""
        result = self.run_pandas_ai(
            initial_df, query, is_conversational_answer=is_conversational_answer
        )
        if is_conversational_answer:
            return [Document(text=result)]
        else:
            if isinstance(result, (np.generic)):
                result = pd.Series(result)
            elif isinstance(result, (pd.Series, pd.DataFrame)):
                pass
            else:
                raise ValueError(f"Unexpected type for result: {type(result)}")
            # if not conversational answer, use Pandas CSV Reader
            reader = PandasCSVReader(
                concat_rows=self._concat_rows,
                col_joiner=self._col_joiner,
                row_joiner=self._row_joiner,
                pandas_config=self._pandas_config,
            )

            with TemporaryDirectory() as tmpdir:
                outpath = Path(tmpdir) / "out.csv"
                with outpath.open("w") as f:
                    # TODO: add option to specify index=False
                    result.to_csv(f, index=False)

                return reader.load_data(outpath)
