"""Base parser and config class."""

from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union


class BaseParser:
    """Base class for all parsers."""

    def __init__(self, parser_config: Optional[Dict] = None):
        """Init params."""
        self._parser_config = parser_config

    def init_parser(self) -> None:
        """Init parser and store it."""
        parser_config = self._init_parser()
        self._parser_config = parser_config

    @property
    def parser_config_set(self) -> bool:
        """Check if parser config is set."""
        return self._parser_config is not None

    @property
    def parser_config(self) -> Dict:
        """Check if parser config is set."""
        if self._parser_config is None:
            raise ValueError("Parser config not set.")
        return self._parser_config

    @abstractmethod
    def _init_parser(self) -> Dict:
        """Initialize the parser with the config."""

    @abstractmethod
    def parse_file(self, file: Path, errors: str = "ignore") -> Union[str, List[str]]:
        """Parse file."""
