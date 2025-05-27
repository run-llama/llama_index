from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fsspec import AbstractFileSystem
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class StructuredDataReader(BaseReader):
    """
    Updated BaseReader parser to support JSON, JSONL, CSV and Excel (.xlsx) files.

    ...

    Args:
        col_joiner (str): The string to join the columns with. Defaults to ', '.
        col_index (str, int, or list): The list of columns to be used as index.
        col_metadata (None, str, int, or list): The list of columns to be used as metadata.
        ...

    """

    def __init__(
        self,
        *args: Any,
        col_joiner: str = ", ",
        pandas_config: dict = {},
        col_index: Union[str, int, List],
        col_metadata: Optional[Union[str, int, List]] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._col_joiner = col_joiner
        self._pandas_config = pandas_config
        self._col_index = col_index
        self._col_metadata = col_metadata

    def _load_dataframe(
        self, file: Path, fs: Optional[AbstractFileSystem] = None
    ) -> pd.DataFrame:
        file_extension = file.suffix.lower()

        read_funcs = {
            ".csv": lambda f: pd.read_csv(f),
            ".xlsx": lambda f: pd.read_excel(f),
            ".json": lambda f: pd.read_json(f, encoding="utf-8"),
            ".jsonl": lambda f: pd.read_json(f, encoding="utf-8", lines=True),
        }

        if file_extension not in read_funcs:
            raise ValueError(
                f"Unsupported file extension '{file_extension}'. Supported extensions are 'json', 'csv', 'xlsx', and 'jsonl'."
            )

        if fs:
            with fs.open(file) as f:
                df = read_funcs[file_extension](f, **self._pandas_config)
        else:
            df = read_funcs[file_extension](file, **self._pandas_config)
        return df

    def _validate_column(self, index_name, column_index, df):
        if isinstance(column_index, int):
            assert -len(df.columns) < column_index < len(df.columns), (
                f"The {index_name} {column_index} exceeds the range of columns in the dataframe: ({len(df.columns)})"
            )
        elif isinstance(column_index, str):
            assert column_index in df.columns, (
                f"The {index_name} must be in the dataframe"
            )
        else:
            if all(isinstance(item, int) for item in column_index):
                assert all(
                    -len(df.columns) < item < len(df.columns) for item in column_index
                ), (
                    f"Some items in {index_name} exceed the range of columns in the dataframe: ({len(df.columns)})"
                )
            elif all(isinstance(item, str) for item in column_index):
                assert set(column_index).issubset(df.columns), (
                    f"All columns in {index_name} must be in the dataframe"
                )
            else:
                raise ValueError(
                    "Not support int and str columns both in column configs."
                )

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        df = self._load_dataframe(file, fs)

        assert self._col_index, f"The col_index must be specified"
        self._validate_column("col_index", self._col_index, df)

        if isinstance(self._col_index, int) or (
            isinstance(self._col_index, list)
            and all(isinstance(item, int) for item in self._col_index)
        ):
            df_text = df.iloc[:, self._col_index]
        else:
            df_text = df[self._col_index]

        if isinstance(df_text, pd.DataFrame):
            text_list = df_text.apply(
                lambda row: self._col_joiner.join(row.astype(str).tolist()), axis=1
            ).tolist()
        elif isinstance(df_text, pd.Series):
            text_list = df_text.tolist()

        if not self._col_metadata:
            return [
                Document(text=text_tuple, metadata=(extra_info or {}))
                for text_tuple in text_list
            ]
        else:
            self._validate_column("col_metadata", self._col_metadata, df)
            if isinstance(self._col_metadata, int) or (
                isinstance(self._col_metadata, list)
                and all(isinstance(item, int) for item in self._col_metadata)
            ):
                df_metadata = df.iloc[:, self._col_metadata]
            else:
                df_metadata = df[self._col_metadata]

            if isinstance(df_metadata, pd.Series):
                df_metadata = pd.DataFrame(df_metadata)

            metadata_list = df_metadata.to_dict(orient="records")

            return [
                Document(
                    text=text_tuple, metadata={**(metadata_tuple), **(extra_info or {})}
                )
                for text_tuple, metadata_tuple in zip(text_list, metadata_list)
            ]
