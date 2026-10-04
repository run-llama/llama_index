from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class IPYNBReader(BaseReader):
    """
    Read code and Markdown from Jupyter notebooks.

    Documents follow notebook cell boundaries unless ``concatenate=True``.
    Code cells do not need to be executed before reading.
    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        concatenate: bool = False,
    ):
        """Init params."""
        self._parser_config = parser_config
        self._concatenate = concatenate

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        try:
            import nbconvert
            import nbformat
            from traitlets.config import Config
        except ImportError:
            raise ImportError("Please install nbconvert 'pip install nbconvert' ")

        if fs:
            with fs.open(file, encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)
        else:
            notebook = nbformat.read(file, as_version=4)

        exporter = nbconvert.exporters.ScriptExporter(
            config=Config({"TemplateExporter": {"exclude_input_prompt": True}})
        )
        cells = notebook.cells
        notebook.cells = []
        header = exporter.from_notebook_node(notebook)[0]
        splits = []
        for cell in cells:
            if not cell.source.strip():
                continue
            # Export actual cells so prompt-like text in their source is preserved.
            notebook.cells = [cell]
            text = exporter.from_notebook_node(notebook)[0].removeprefix(header)
            if (
                cell.cell_type == "markdown"
                and not text.strip()
                and not cell.metadata.get("transient", {}).get("remove_source", False)
            ):
                text = cell.source
            if text.strip():
                splits.append(text)

        if self._concatenate:
            docs = [Document(text="\n\n".join(splits), metadata=extra_info or {})]
        else:
            docs = [Document(text=s, metadata=extra_info or {}) for s in splits]
        return docs
