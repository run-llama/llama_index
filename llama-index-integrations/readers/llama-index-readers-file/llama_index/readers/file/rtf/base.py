"""RTF (Rich Text Format) reader."""
from pathlib import Path
from typing import List, Union, Any, Dict, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class RTFReader(BaseReader):
    """RTF (Rich Text Format) Reader. Reads rtf file and convert to Document."""

    def load_data(
        self,
        input_file: Union[Path, str],
        extra_info: Optional[Dict[str, Any]] = None,
        **load_kwargs: Any
    ) -> List[Document]:
        """Load data from RTF file.

        Args:
            input_file (Path | str): Path for the RTF file.
            extra_info (Dict[str, Any]): Path for the RTF file.

        Returns:
            List[Document]: List of documents.
        """
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            raise ImportError("striprtf is required to read RTF files.")

        with open(str(input_file)) as f:
            text = rtf_to_text(f.read())
            return [Document(text=text.strip(), metadata=extra_info or {})]
