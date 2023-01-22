"""Tabular parser.

Contains parsers for tabular data files.

"""
from pathlib import Path
from typing import Dict

from gpt_index.readers.file.base_parser import BaseParser


class CSVParser(BaseParser):
    """CSV parser."""

    def _init_parser(self) -> Dict:
        """Init parser."""
        return {}

    def parse_file(self, file: Path, errors: str = "ignore") -> str:
        """Parse file."""
        try:
            import csv
        except ImportError:
            raise ValueError("csv module is required to read CSV files.")
        text_list = []
        with open(file, "r") as fp:
            csv_reader = csv.reader(fp)
            for row in csv_reader:
                text_list.append(", ".join(row))
        text = "\n".join(text_list)

        return text
