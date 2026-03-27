"""CRW (Firecrawl-compatible) reader. Install ``llama-index-readers-web``."""

from llama_index.readers.web.crw_web.base import VALID_MODES, CrwWebReader


class CrwReader(CrwWebReader):
    """
    Load web pages via a self-hosted CRW server (https://github.com/us/crw).

    Defaults to ``http://localhost:3000``. Use ``mode`` on the reader or pass
    ``mode`` to :meth:`load_data` (``scrape``, ``crawl``, or ``map``).
    """

    @classmethod
    def class_name(cls) -> str:
        return "CrwReader"


__all__ = ["CrwReader", "VALID_MODES", "CrwWebReader"]
