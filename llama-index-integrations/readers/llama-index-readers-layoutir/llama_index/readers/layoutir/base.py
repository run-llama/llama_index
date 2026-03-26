import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from layoutir import Pipeline
from layoutir.adapters import DoclingAdapter
from layoutir.chunking import SemanticSectionChunker
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from pydantic import Field


class LayoutIRReader(BasePydanticReader):
    """
    LayoutIR Reader.

    Production-grade document ingestion engine using LayoutIR's compiler-like architecture.
    Processes PDFs and documents through IR (Intermediate Representation) to preserve
    complex layouts, tables, and multi-column structures.

    Args:
        use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
        api_key (Optional[str], optional): API key for remote processing. Defaults to None.
        model_name (Optional[str], optional): Model name to use for processing. Defaults to None.
        chunk_strategy (str, optional): Chunking strategy to use. Options: "semantic", "fixed". Defaults to "semantic".
        max_heading_level (int, optional): Maximum heading level for semantic chunking. Defaults to 2.

    """

    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU acceleration for document processing.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for remote LayoutIR processing.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name to use for document processing.",
    )
    chunk_strategy: str = Field(
        default="semantic",
        description="Chunking strategy: 'semantic' for section-based, 'fixed' for fixed-size chunks.",
    )
    max_heading_level: int = Field(
        default=2,
        description="Maximum heading level for semantic chunking.",
    )
    is_remote: bool = Field(
        default=False,
        description="Whether the data is loaded from a remote API or a local file.",
    )

    def lazy_load_data(
        self,
        file_path: Union[str, Path, List[str], List[Path]],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Document]:
        """
        Lazily load documents from given file path(s) using LayoutIR.

        Args:
            file_path (Union[str, Path, List[str], List[Path]]): Path to PDF/document file(s).
            extra_info (Optional[Dict[str, Any]], optional): Additional metadata to include. Defaults to None.

        Yields:
            Document: LlamaIndex Document objects with preserved layout structure.

        Raises:
            ImportError: If GPU is requested but PyTorch is not installed.

        """
        # Check GPU requirements if use_gpu is enabled
        if self.use_gpu:
            try:
                import torch  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "GPU acceleration requested but PyTorch is not installed. "
                    "Please install PyTorch with CUDA support:\n"
                    "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
                ) from e

        # Normalize file_path to list
        file_paths = file_path if isinstance(file_path, list) else [file_path]

        # Initialize LayoutIR components
        adapter_kwargs = {"use_gpu": self.use_gpu}
        if self.model_name:
            adapter_kwargs["model_name"] = self.model_name
        if self.api_key:
            adapter_kwargs["api_key"] = self.api_key

        adapter = DoclingAdapter(**adapter_kwargs)

        # Setup chunking strategy
        if self.chunk_strategy == "semantic":
            chunker = SemanticSectionChunker(max_heading_level=self.max_heading_level)
        else:
            chunker = None  # Use default chunking

        pipeline = Pipeline(adapter=adapter, chunk_strategy=chunker)

        # Process each file
        for source in file_paths:
            source_path = Path(source) if isinstance(source, str) else source

            # Use a temp directory for LayoutIR output, cleaned up after processing
            tmp_dir = tempfile.mkdtemp()
            try:
                layoutir_doc = pipeline.process(
                    input_path=source_path,
                    output_dir=Path(tmp_dir),
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Extract blocks/chunks from the IR
            if hasattr(layoutir_doc, "blocks"):
                blocks = layoutir_doc.blocks
            elif hasattr(layoutir_doc, "chunks"):
                blocks = layoutir_doc.chunks
            else:
                # Fallback: treat entire document as single block
                blocks = [{"text": str(layoutir_doc), "type": "document"}]

            # Convert each block to a LlamaIndex Document
            for idx, block in enumerate(blocks):
                # Extract text content from layoutir.schema.Block objects
                if isinstance(block, dict):
                    text = block.get("text", block.get("content", ""))
                    block_type = str(block.get("type", "unknown"))
                    block_id = block.get("id", f"{source_path.stem}_block_{idx}")
                    page_number = block.get("page", block.get("page_number", 0))
                elif hasattr(block, "content"):
                    text = block.content or ""
                    block_type = (
                        str(block.type.value)
                        if hasattr(block.type, "value")
                        else str(block.type)
                    )
                    block_id = getattr(
                        block, "block_id", f"{source_path.stem}_block_{idx}"
                    )
                    page_number = getattr(block, "page_number", 0)
                else:
                    text = str(block)
                    block_type = "block"
                    block_id = f"{source_path.stem}_block_{idx}"
                    page_number = 0

                # Create metadata
                metadata = extra_info.copy() if extra_info else {}
                metadata.update(
                    {
                        "file_path": str(source_path),
                        "file_name": source_path.name,
                        "block_type": block_type,
                        "block_index": idx,
                        "page_number": page_number,
                        "source": "layoutir",
                    }
                )

                # Create and yield Document
                doc = Document(
                    doc_id=block_id,
                    text=text,
                    metadata=metadata,
                )

                yield doc
