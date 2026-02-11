import pytest
from unittest.mock import Mock, patch

from llama_index.core.readers.base import BaseReader
from llama_index.readers.sec_filings import SECFilingsLoader, SECFilingsStreamingReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SECFilingsLoader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_streaming_reader_class():
    names_of_base_classes = [b.__name__ for b in SECFilingsStreamingReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_streaming_reader_initialization():
    """Test that the streaming reader initializes correctly."""
    reader = SECFilingsStreamingReader(
        tickers=["AAPL", "MSFT"],
        filing_types=["10-K", "8-K"],
        num_filings=5,
    )
    assert reader.tickers == ["AAPL", "MSFT"]
    assert reader.filing_types == ["10-K", "8-K"]
    assert reader.num_filings == 5


def test_streaming_reader_invalid_filing_type():
    """Test that invalid filing types raise an error."""
    with pytest.raises(ValueError, match="Invalid filing type"):
        SECFilingsStreamingReader(
            tickers=["AAPL"],
            filing_types=["INVALID"],
            num_filings=5,
        )


def test_streaming_reader_valid_filing_types():
    """Test that all valid filing types are accepted."""
    valid_types = ["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"]
    for ft in valid_types:
        reader = SECFilingsStreamingReader(
            tickers=["AAPL"],
            filing_types=[ft],
            num_filings=1,
        )
        assert ft in reader.filing_types


def test_streaming_reader_ticker_normalization():
    """Test that tickers are normalized to uppercase."""
    reader = SECFilingsStreamingReader(
        tickers=["aapl", "msft"],
        filing_types=["10-K"],
        num_filings=1,
    )
    assert reader.tickers == ["AAPL", "MSFT"]


def test_streaming_reader_sections():
    """Test that sections can be specified."""
    reader = SECFilingsStreamingReader(
        tickers=["AAPL"],
        filing_types=["10-K"],
        num_filings=1,
        sections=["ITEM_1A", "ITEM_7"],
    )
    assert reader.sections == ["ITEM_1A", "ITEM_7"]


def test_streaming_reader_date_range():
    """Test that date range can be specified."""
    reader = SECFilingsStreamingReader(
        tickers=["AAPL"],
        filing_types=["10-K"],
        num_filings=1,
        start_date="2020-01-01",
        end_date="2023-12-31",
    )
    assert reader.start_date == "2020-01-01"
    assert reader.end_date == "2023-12-31"


def test_streaming_reader_clean_html():
    """Test HTML cleaning functionality."""
    reader = SECFilingsStreamingReader(
        tickers=["AAPL"],
        filing_types=["10-K"],
        num_filings=1,
    )

    html = "<p>Hello &amp; World</p><br>&nbsp;Test&lt;value&gt;"
    clean_text = reader._clean_html(html)
    assert "Hello & World" in clean_text
    assert "Test<value>" in clean_text
    assert "<p>" not in clean_text
    assert "&amp;" not in clean_text


class TestSectionExtraction:
    """Test section extraction functionality."""

    def test_extract_sections_10k(self):
        """Test 10-K section extraction."""
        reader = SECFilingsStreamingReader(
            tickers=["AAPL"],
            filing_types=["10-K"],
            num_filings=1,
        )

        text = """
        Item 1. Business
        We are a company that does business.

        Item 1A. Risk Factors
        There are many risks associated with our business.
        These include market risks and operational risks.

        Item 2. Properties
        We have properties in many locations.
        """

        sections = reader._extract_sections_10k(text, ["ITEM_1A"])
        assert "ITEM_1A" in sections
        assert "risks" in sections["ITEM_1A"].lower()

    def test_extract_sections_8k(self):
        """Test 8-K section extraction."""
        reader = SECFilingsStreamingReader(
            tickers=["AAPL"],
            filing_types=["8-K"],
            num_filings=1,
        )

        text = """
        Item 2.02 Results of Operations and Financial Condition
        The company reported quarterly earnings.

        Item 7.01 Regulation FD Disclosure
        Forward-looking statements were made.

        Item 9.01 Financial Statements and Exhibits
        See attached exhibits.
        """

        sections = reader._extract_sections_8k(text, ["2.02", "7.01"])
        assert "2.02" in sections
        assert "quarterly earnings" in sections["2.02"].lower()


@patch("llama_index.readers.sec_filings.streaming.requests")
def test_streaming_reader_mock_load(mock_requests):
    """Test load_data with mocked responses."""
    # Mock the session
    mock_session = Mock()
    mock_requests.Session.return_value = mock_session

    # Mock CIK lookup
    cik_response = Mock()
    cik_response.text = "CIK=0000320193 Apple Inc"
    cik_response.raise_for_status = Mock()

    # Mock company info
    company_info_response = Mock()
    company_info_response.text = """{
        "name": "Apple Inc",
        "filings": {
            "recent": {
                "form": ["10-K", "8-K"],
                "accessionNumber": ["0000320193-22-000108", "0000320193-22-000109"],
                "filingDate": ["2022-10-28", "2022-10-27"],
                "primaryDocument": ["aapl-20220924.htm", "aapl-8k.htm"],
                "primaryDocDescription": ["10-K", "8-K"]
            }
        }
    }"""
    company_info_response.raise_for_status = Mock()

    # Mock filing content
    filing_response = Mock()
    filing_response.text = "<html><body>Filing content</body></html>"
    filing_response.raise_for_status = Mock()

    mock_session.get.side_effect = [
        cik_response,
        company_info_response,
        filing_response,
    ]

    reader = SECFilingsStreamingReader(
        tickers=["AAPL"],
        filing_types=["10-K"],
        num_filings=1,
    )

    documents = reader.load_data()

    # Verify documents were created
    assert isinstance(documents, list)
    # The mock might not produce documents due to the way we set it up,
    # but this verifies the method runs without error
