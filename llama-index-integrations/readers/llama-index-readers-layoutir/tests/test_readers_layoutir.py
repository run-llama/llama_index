from unittest.mock import MagicMock, patch

import pytest
from llama_index.readers.layoutir.base import LayoutIRReader


@pytest.fixture
def mock_layoutir_components(monkeypatch):
    """Mock all layoutir components to avoid external dependencies."""
    # Mock the imports
    mock_pipeline = MagicMock()
    mock_adapter = MagicMock()
    mock_chunker = MagicMock()

    mock_modules = {
        "layoutir": MagicMock(),
        "layoutir.Pipeline": mock_pipeline,
        "layoutir.adapters": MagicMock(DoclingAdapter=mock_adapter),
        "layoutir.chunking": MagicMock(SemanticSectionChunker=mock_chunker),
    }

    # Patch sys.modules to mock layoutir
    import sys

    for module_name, module_mock in mock_modules.items():
        sys.modules[module_name] = module_mock

    yield {
        "pipeline": mock_pipeline,
        "adapter": mock_adapter,
        "chunker": mock_chunker,
    }

    # Cleanup
    for module_name in mock_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_layoutir_reader_initialization():
    """Test LayoutIRReader can be initialized with default parameters."""
    reader = LayoutIRReader()

    assert reader.use_gpu is False
    assert reader.api_key is None
    assert reader.model_name is None
    assert reader.chunk_strategy == "semantic"
    assert reader.max_heading_level == 2
    assert reader.is_remote is False


def test_layoutir_reader_initialization_with_custom_params():
    """Test LayoutIRReader can be initialized with custom parameters."""
    reader = LayoutIRReader(
        use_gpu=True,
        api_key="test-key",
        model_name="custom-model",
        chunk_strategy="fixed",
        max_heading_level=3,
    )

    assert reader.use_gpu is True
    assert reader.api_key == "test-key"
    assert reader.model_name == "custom-model"
    assert reader.chunk_strategy == "fixed"
    assert reader.max_heading_level == 3


def test_lazy_load_data_single_file(monkeypatch):
    """Test loading a single file with mocked LayoutIR pipeline."""
    # Create mock document with blocks
    mock_doc = MagicMock()
    mock_doc.blocks = [
        {
            "text": "First paragraph content",
            "type": "paragraph",
            "id": "block_0",
            "page": 1,
        },
        {
            "text": "Second paragraph content",
            "type": "paragraph",
            "id": "block_1",
            "page": 1,
        },
        {"text": "Table content", "type": "table", "id": "block_2", "page": 2},
    ]

    # Create mock pipeline
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.process.return_value = mock_doc

    # Mock the Pipeline class
    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)

    # Mock the adapter and chunker classes
    mock_adapter_class = MagicMock()
    mock_chunker_class = MagicMock()

    # Patch the imports in the base module
    with patch("llama_index.readers.layoutir.base.Pipeline", mock_pipeline_class):
        with patch(
            "llama_index.readers.layoutir.base.DoclingAdapter", mock_adapter_class
        ):
            with patch(
                "llama_index.readers.layoutir.base.SemanticSectionChunker",
                mock_chunker_class,
            ):
                reader = LayoutIRReader()
                documents = list(reader.lazy_load_data(file_path="test.pdf"))

    # Verify we got 3 documents (one per block)
    assert len(documents) == 3

    # Check first document
    assert documents[0].text == "First paragraph content"
    assert documents[0].metadata["block_type"] == "paragraph"
    assert documents[0].metadata["page_number"] == 1
    assert documents[0].metadata["block_index"] == 0
    assert documents[0].metadata["source"] == "layoutir"
    assert documents[0].doc_id == "block_0"

    # Check second document
    assert documents[1].text == "Second paragraph content"
    assert documents[1].metadata["block_type"] == "paragraph"

    # Check third document (table)
    assert documents[2].text == "Table content"
    assert documents[2].metadata["block_type"] == "table"
    assert documents[2].metadata["page_number"] == 2


def test_lazy_load_data_multiple_files(monkeypatch):
    """Test loading multiple files."""
    # Create mock document
    mock_doc = MagicMock()
    mock_doc.blocks = [
        {"text": "Content", "type": "paragraph", "id": "block_0", "page": 1}
    ]

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.process.return_value = mock_doc

    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)
    mock_adapter_class = MagicMock()
    mock_chunker_class = MagicMock()

    with patch("llama_index.readers.layoutir.base.Pipeline", mock_pipeline_class):
        with patch(
            "llama_index.readers.layoutir.base.DoclingAdapter", mock_adapter_class
        ):
            with patch(
                "llama_index.readers.layoutir.base.SemanticSectionChunker",
                mock_chunker_class,
            ):
                reader = LayoutIRReader()
                documents = list(
                    reader.lazy_load_data(file_path=["file1.pdf", "file2.pdf"])
                )

    # Should get 2 documents (1 block per file)
    assert len(documents) == 2
    assert mock_pipeline_instance.process.call_count == 2


def test_lazy_load_data_with_extra_info(monkeypatch):
    """Test that extra_info metadata is preserved."""
    mock_doc = MagicMock()
    mock_doc.blocks = [
        {"text": "Content", "type": "paragraph", "id": "block_0", "page": 1}
    ]

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.process.return_value = mock_doc

    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)
    mock_adapter_class = MagicMock()
    mock_chunker_class = MagicMock()

    with patch("llama_index.readers.layoutir.base.Pipeline", mock_pipeline_class):
        with patch(
            "llama_index.readers.layoutir.base.DoclingAdapter", mock_adapter_class
        ):
            with patch(
                "llama_index.readers.layoutir.base.SemanticSectionChunker",
                mock_chunker_class,
            ):
                reader = LayoutIRReader()
                extra_metadata = {"department": "research", "year": 2026}
                documents = list(
                    reader.lazy_load_data(
                        file_path="test.pdf", extra_info=extra_metadata
                    )
                )

    # Check that extra metadata is included
    assert documents[0].metadata["department"] == "research"
    assert documents[0].metadata["year"] == 2026
    # Standard metadata should also be present
    assert documents[0].metadata["block_type"] == "paragraph"
    assert documents[0].metadata["source"] == "layoutir"


def test_lazy_load_data_with_gpu(monkeypatch):
    """Test that GPU flag is passed to adapter."""
    mock_doc = MagicMock()
    mock_doc.blocks = [
        {"text": "Content", "type": "paragraph", "id": "block_0", "page": 1}
    ]

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.process.return_value = mock_doc

    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)
    mock_adapter_class = MagicMock()
    mock_chunker_class = MagicMock()

    with patch("llama_index.readers.layoutir.base.Pipeline", mock_pipeline_class):
        with patch(
            "llama_index.readers.layoutir.base.DoclingAdapter", mock_adapter_class
        ):
            with patch(
                "llama_index.readers.layoutir.base.SemanticSectionChunker",
                mock_chunker_class,
            ):
                reader = LayoutIRReader(use_gpu=True, api_key="test-key")
                documents = list(reader.lazy_load_data(file_path="test.pdf"))

    # Verify adapter was called with GPU flag
    mock_adapter_class.assert_called_once()
    call_kwargs = mock_adapter_class.call_args[1]
    assert call_kwargs["use_gpu"] is True
    assert call_kwargs["api_key"] == "test-key"


def test_lazy_load_data_with_chunks_attribute(monkeypatch):
    """Test fallback when document has chunks instead of blocks."""
    mock_doc = MagicMock()
    # Remove blocks attribute, add chunks instead
    del mock_doc.blocks
    mock_doc.chunks = [{"text": "Chunk 1", "type": "chunk", "id": "chunk_0", "page": 1}]

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.process.return_value = mock_doc

    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)
    mock_adapter_class = MagicMock()
    mock_chunker_class = MagicMock()

    with patch("llama_index.readers.layoutir.base.Pipeline", mock_pipeline_class):
        with patch(
            "llama_index.readers.layoutir.base.DoclingAdapter", mock_adapter_class
        ):
            with patch(
                "llama_index.readers.layoutir.base.SemanticSectionChunker",
                mock_chunker_class,
            ):
                reader = LayoutIRReader()
                documents = list(reader.lazy_load_data(file_path="test.pdf"))

    assert len(documents) == 1
    assert documents[0].text == "Chunk 1"


def test_lazy_load_data_gpu_check_error():
    """Test that ImportError is raised when GPU is requested but requirements are not met."""
    reader = LayoutIRReader(use_gpu=True)

    # Mock torch import to raise ImportError
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'torch'")
    ):
        with pytest.raises(ImportError) as exc_info:
            list(reader.lazy_load_data(file_path="test.pdf"))

        assert "GPU acceleration requested" in str(exc_info.value)
        assert "PyTorch" in str(exc_info.value)


def test_load_data_method(monkeypatch):
    """Test that load_data returns a list instead of iterator."""
    mock_doc = MagicMock()
    mock_doc.blocks = [
        {"text": "Content", "type": "paragraph", "id": "block_0", "page": 1}
    ]

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.process.return_value = mock_doc

    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)
    mock_adapter_class = MagicMock()
    mock_chunker_class = MagicMock()

    with patch("llama_index.readers.layoutir.base.Pipeline", mock_pipeline_class):
        with patch(
            "llama_index.readers.layoutir.base.DoclingAdapter", mock_adapter_class
        ):
            with patch(
                "llama_index.readers.layoutir.base.SemanticSectionChunker",
                mock_chunker_class,
            ):
                reader = LayoutIRReader()
                documents = reader.load_data(file_path="test.pdf")

    # Should return a list, not an iterator
    assert isinstance(documents, list)
    assert len(documents) == 1
