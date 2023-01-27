"""Tabular parser.

Contains parsers for tabular data files.

"""
from pathlib import Path
from typing import Any, Dict, List, Union

from gpt_index.readers.file.base_parser import BaseParser


class CSVParser(BaseParser):
    """CSV parser."""

    def __init__(self, *args: Any, concat_rows: bool = True, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows

    def _init_parser(self) -> Dict:
        """Init parser."""
        return {}

    def parse_file(self, file: Path, errors: str = "ignore") -> Union[str, List[str]]:
        """Parse file.

        Args:
            concatenate (bool): whether to concatenate all rows into one document.
                If set to False, a Document will be created for each row.
                True by default.

        Returns:
            Union[str, List[str]]: a string or a List of strings.

        """
        try:
            import csv
        except ImportError:
            raise ValueError("csv module is required to read CSV files.")
        text_list = []
        with open(file, "r") as fp:
            csv_reader = csv.reader(fp)
            for row in csv_reader:
                text_list.append(", ".join(row))
        if self._concat_rows:
            return "\n".join(text_list)
        else:
            return text_list
