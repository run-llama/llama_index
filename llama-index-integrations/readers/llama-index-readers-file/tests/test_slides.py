import pytest
from llama_index.readers.file.slides import PptxReader
from .generate_test_ppt import create_comprehensive_test_presentation


@pytest.fixture()
def pptx_file(tmp_path):
    """Create a temporary PowerPoint file for testing."""
    if create_comprehensive_test_presentation is None:
        pytest.skip("generate_test_ppt not available")

    # Create test presentation in temp directory
    file_path = tmp_path / "test_presentation.pptx"
    create_comprehensive_test_presentation(str(file_path))
    return file_path


def test_pptx_reader_init():
    """Test PptxReader initialization."""
    reader = PptxReader(extract_images=False, num_workers=2)
    assert reader.extract_images is False
    assert reader.num_workers == 2


def test_load_data_pptx(pptx_file):
    """Test loading PowerPoint data."""
    reader = PptxReader(extract_images=False, context_consolidation_with_llm=False)

    documents = reader.load_data(pptx_file)

    # Basic validation
    assert len(documents) == 12  # Should have 12 slides
    assert all(hasattr(doc, "text") for doc in documents)
    assert all(hasattr(doc, "metadata") for doc in documents)

    # Check first slide
    first_slide = documents[0]
    assert "Enhanced PowerPoint Reader Test" in first_slide.text
    assert first_slide.metadata["page_label"] == 1
    assert len(first_slide.metadata["notes"]) > 0  # Check notes content exists


def test_table_extraction(pptx_file):
    """Test table data extraction."""
    reader = PptxReader()
    documents = reader.load_data(pptx_file)

    # Find slides with tables
    table_slides = [doc for doc in documents if len(doc.metadata.get("tables", [])) > 0]
    assert len(table_slides) >= 1

    # Check table metadata
    table_slide = table_slides[0]
    tables = table_slide.metadata.get("tables", [])
    assert len(tables) > 0

    table = tables[0]
    assert "headers" in table
    assert "data" in table
    assert "dimensions" in table


def test_chart_extraction(pptx_file):
    """Test chart metadata extraction."""
    reader = PptxReader()
    documents = reader.load_data(pptx_file)

    # Find slides with charts
    chart_slides = [doc for doc in documents if len(doc.metadata.get("charts", [])) > 0]
    assert len(chart_slides) >= 1

    # Check chart metadata
    chart_slide = chart_slides[0]
    charts = chart_slide.metadata.get("charts", [])
    assert len(charts) > 0

    chart = charts[0]
    assert "chart_type" in chart
    assert "series_info" in chart


def test_speaker_notes_extraction(pptx_file):
    """Test speaker notes extraction."""
    reader = PptxReader()
    documents = reader.load_data(pptx_file)

    # All slides should have notes
    slides_with_notes = [
        doc for doc in documents if len(doc.metadata.get("notes", "")) > 0
    ]
    assert len(slides_with_notes) == 12

    # Check notes content
    first_slide = documents[0]
    notes = first_slide.metadata.get("notes", "")
    assert len(notes) > 0
    assert "comprehensive test presentation" in notes.lower()


def test_content_consolidation(pptx_file):
    """Test content consolidation structure."""
    reader = PptxReader()
    documents = reader.load_data(pptx_file)

    # Check content structure
    for doc in documents:
        assert "-----" in doc.text  # Section separators
        assert len(doc.text) > 0  # Content should exist


def test_multithreading(pptx_file):
    """Test multithreaded processing."""
    reader = PptxReader(num_workers=2, batch_size=4)
    documents = reader.load_data(pptx_file)

    # Should process successfully with threading
    assert len(documents) == 12
    assert all(doc.metadata.get("page_label") for doc in documents)


def test_llm_consolidation_with_settings_llm(pptx_file):
    """Test LLM consolidation when LLM is set in Settings but not passed directly."""
    from llama_index.core import Settings
    from llama_index.core.llms.mock import MockLLM

    Settings.llm = MockLLM()
    reader = PptxReader(
        extract_images=False,
        context_consolidation_with_llm=True,  # Request LLM consolidation
        llm=None,  # Don't pass LLM directly
        num_workers=2,
    )

    # Should still return results
    documents = reader.load_data(pptx_file)

    # Basic validation
    assert len(documents) == 12
    assert all(hasattr(doc, "text") for doc in documents)


def test_llm_consolidation_with_direct_llm(pptx_file):
    """Test LLM consolidation when LLM is passed directly to PptxReader."""
    from llama_index.core.llms.mock import MockLLM

    # Create MockLLM directly and pass it to the reader
    mock_llm = MockLLM()

    reader = PptxReader(
        extract_images=False,
        context_consolidation_with_llm=True,  # Request LLM consolidation
        llm=mock_llm,  # Pass LLM directly
        num_workers=2,
    )

    # Should use the directly passed LLM
    assert reader.context_consolidation_with_llm is True
    assert reader.llm is mock_llm  # Should be the exact same instance

    # Should process successfully
    documents = reader.load_data(pptx_file)

    # Basic validation
    assert len(documents) == 12
    assert all(hasattr(doc, "text") for doc in documents)
    assert all(hasattr(doc, "metadata") for doc in documents)

    # Content should be consolidated
    for doc in documents:
        assert "-----" in doc.text  # Section separators should be there
        assert len(doc.text) > 0  # Should have content
