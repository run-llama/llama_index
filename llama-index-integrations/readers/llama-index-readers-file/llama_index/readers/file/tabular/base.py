"""Tabular parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
import importlib

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class CSVReader(BaseReader):
    """CSV parser.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

    """

    def __init__(self, *args: Any, concat_rows: bool = True, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file.

        Returns:
            Union[str, List[str]]: a string or a List of strings.

        """
        try:
            import csv
        except ImportError:
            raise ImportError("csv module is required to read CSV files.")
        text_list = []
        with open(file) as fp:
            csv_reader = csv.reader(fp)
            for row in csv_reader:
                text_list.append(", ".join(row))

        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        if self._concat_rows:
            return [Document(text="\n".join(text_list), metadata=metadata)]
        else:
            return [Document(text=text, metadata=metadata) for text in text_list]


class PandasCSVReader(BaseReader):
    r"""Pandas-based CSV parser.

    Parses CSVs using the separator detection from Pandas `read_csv`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:
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
        *args: Any,
        concat_rows: bool = True,
        col_joiner: str = ", ",
        row_joiner: str = "\n",
        pandas_config: dict = {},
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)

        text_list = df.apply(
            lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        ).tolist()

        if self._concat_rows:
            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=extra_info or {}
                )
            ]
        else:
            return [
                Document(text=text, metadata=extra_info or {}) for text in text_list
            ]


class PandasExcelReader(BaseReader):
    """Custom Excel parser that includes header names in each row.

    Parses Excel files using Pandas' `read_excel` function, but formats
    each row to include the header name, for example: "name: joao, position: analyst".
    The first row (header) is not included in the generated documents.

    Args:
        concat_rows (bool): Determines whether to concatenate all rows into one document.
            If set to False, one Document is created for each row.
            Defaults to True.
        sheet_name (str | int | None): Defaults to None, meaning all sheets.
            Alternatively, pass a string or an integer to specify the sheet to be read.
        field_separator (str): Character or string to separate each field. Default: ", ".
        key_value_separator (str): Character or string to separate the key from the value. Default: ": ".
        pandas_config (dict): Options for the `pandas.read_excel` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
            for more details.
            Defaults to an empty dictionary.
    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        sheet_name=None,
        field_separator: str = ", ",
        key_value_separator: str = ": ",
        pandas_config: dict = {},
        **kwargs: Any,
    ) -> None:
        """Initializes the parameters."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._sheet_name = sheet_name
        self._field_separator = field_separator
        self._key_value_separator = key_value_separator
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parses the file."""
        openpyxl_spec = importlib.util.find_spec("openpyxl")
        if openpyxl_spec is not None:
            pass
        else:
            raise ImportError(
                "Please install openpyxl to read Excel files. You can install it with 'pip install openpyxl'"
            )

        # A sheet_name of None means all sheets; otherwise, indexing starts at 0
        if fs:
            with fs.open(file) as f:
                dfs = pd.read_excel(f, self._sheet_name, **self._pandas_config)
        else:
            dfs = pd.read_excel(file, self._sheet_name, **self._pandas_config)

        documents = []

        # Handle the case where only a single DataFrame is returned
        if isinstance(dfs, pd.DataFrame):
            df = dfs.fillna("")
            # Get the headers/column names
            headers = df.columns.tolist()

            # Convert the DataFrame into a list of rows formatted with header names
            text_list = []

            # Start from index 0 to include all data rows
            # The header is already in 'headers', not in the data rows
            for _, row in df.iterrows():
                # Format each row as "header1: value1, header2: value2, ..."
                formatted_row = self._field_separator.join(
                    [
                        f"{header}{self._key_value_separator}{row[header]!s}"
                        for header in headers
                    ]
                )
                text_list.append(formatted_row)

            if self._concat_rows:
                documents.append(
                    Document(text="\n".join(text_list), metadata=extra_info or {})
                )
            else:
                documents.extend(
                    [
                        Document(text=text, metadata=extra_info or {})
                        for text in text_list
                    ]
                )
        else:
            # Handle multiple sheets
            for df in dfs.values():
                df = df.fillna("")
                headers = df.columns.tolist()

                text_list = []
                for _, row in df.iterrows():
                    formatted_row = self._field_separator.join(
                        [
                            f"{header}{self._key_value_separator}{row[header]!s}"
                            for header in headers
                        ]
                    )
                    text_list.append(formatted_row)

                if self._concat_rows:
                    documents.append(
                        Document(text="\n".join(text_list), metadata=extra_info or {})
                    )
                else:
                    documents.extend(
                        [
                            Document(text=text, metadata=extra_info or {})
                            for text in text_list
                        ]
                    )

        return documents
