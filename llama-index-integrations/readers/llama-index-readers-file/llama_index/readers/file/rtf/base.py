"""RTF (Rich Text Format) reader."""
import os.path
from pathlib import Path
from typing import List, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class RTFReader(BaseReader):
    """RTF (Rich Text Format) Reader. Reads rtf file and convert to Document."""

    def load_data(self, file: Union[Path, str]) -> List[Document]:
        """Load data from RTF file.

        Args:
            file (Path | str): Path for the RTF file.

        Returns:
            List[Document]: List of documents.
        """
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            raise ImportError("striprtf is required to read RTF files.")

        with open(str(file), "r") as f:
            text = rtf_to_text(f.read())
            file_name = os.path.basename(file)

            return [Document(text=text.strip(), metadata={"filename": file_name, 'file_path': str(file)})]
