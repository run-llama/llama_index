from unittest.mock import Mock, patch
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.valyu import ValyuToolSpec, ValyuRetriever
from llama_index.core.schema import Document, NodeWithScore, TextNode


def test_class():
    names_of_base_classes = [b.__name__ for b in ValyuToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@patch("valyu.Valyu")
def test_init(mock_valyu):
    """Test ValyuToolSpec initialization."""
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    tool = ValyuToolSpec(
        api_key="test_key",
        verbose=True,
        max_price=50.0,
        relevance_threshold=0.8,
        fast_mode=True,
    )

    assert tool.client == mock_client
    assert tool._verbose is True
    assert tool._max_price == 50.0
    assert tool._relevance_threshold == 0.8
    assert tool._fast_mode is True
    mock_valyu.assert_called_once_with(api_key="test_key")


@patch("valyu.Valyu")
def test_init_with_user_controlled_params(mock_valyu):
    """Test ValyuToolSpec initialization with user-controlled search parameters."""
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    tool = ValyuToolSpec(
        api_key="test_key",
        max_price=75.0,
        relevance_threshold=0.7,
        included_sources=["example.com", "trusted.org"],
        excluded_sources=["spam.com"],
        response_length="large",
        country_code="US",
        fast_mode=None,  # Let model decide
    )

    assert tool._max_price == 75.0
    assert tool._relevance_threshold == 0.7
    assert tool._included_sources == ["example.com", "trusted.org"]
    assert tool._excluded_sources == ["spam.com"]
    assert tool._response_length == "large"
    assert tool._country_code == "US"
    assert tool._fast_mode is None


@patch("valyu.Valyu")
def test_search_uses_user_defaults(mock_valyu):
    """Test search method uses user's configured defaults."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result object
    mock_result = Mock()
    mock_result.content = "Test content"
    mock_result.title = "Test title"
    mock_result.url = "https://test.com"
    mock_result.source = "test_source"
    mock_result.price = 1.0
    mock_result.length = 100
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.8

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(
        api_key="test_key",
        max_price=75.0,
        relevance_threshold=0.8,
        included_sources=["trusted.com"],
        excluded_sources=["spam.com"],
        response_length="medium",
        country_code="GB",
        fast_mode=True,
    )

    # Test search uses all user defaults
    documents = tool.search(query="test query")

    # Verify the client was called with user's configured values
    mock_client.search.assert_called_once_with(
        query="test query",
        search_type="all",
        max_num_results=5,
        relevance_threshold=0.8,  # User's setting
        max_price=75.0,  # User's setting
        start_date=None,
        end_date=None,
        included_sources=["trusted.com"],  # User's setting
        excluded_sources=["spam.com"],  # User's setting
        response_length="medium",  # User's setting
        country_code="GB",  # User's setting
        fast_mode=True,  # User's setting
    )

    # Verify document creation
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].text == "Test content"


@patch("valyu.Valyu")
def test_search_model_can_control_allowed_params(mock_valyu):
    """Test search method where model can control specific parameters."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result object
    mock_result = Mock()
    mock_result.content = "Model controlled test"
    mock_result.title = "Model test"
    mock_result.url = "https://model-test.com"
    mock_result.source = "model_source"
    mock_result.price = 2.0
    mock_result.length = 200
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.9

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    # User allows model to control fast_mode
    tool = ValyuToolSpec(
        api_key="test_key",
        max_price=50.0,
        relevance_threshold=0.6,
        fast_mode=None,  # Let model decide
    )

    # Model controls allowed parameters
    documents = tool.search(
        query="model controlled query",
        search_type="proprietary",
        max_num_results=10,
        start_date="2024-01-01",
        end_date="2024-12-31",
        fast_mode=False,  # Model can decide since user set None
    )

    # Verify the client was called with mix of user and model parameters
    mock_client.search.assert_called_once_with(
        query="model controlled query",  # Model
        search_type="proprietary",  # Model
        max_num_results=10,  # Model
        relevance_threshold=0.6,  # User (model cannot override)
        max_price=50.0,  # User (model cannot override)
        start_date="2024-01-01",  # Model
        end_date="2024-12-31",  # Model
        included_sources=None,  # User default (model cannot override)
        excluded_sources=None,  # User default (model cannot override)
        response_length=None,  # User default (model cannot override)
        country_code=None,  # User default (model cannot override)
        fast_mode=False,  # Model (allowed since user set None)
    )


@patch("valyu.Valyu")
def test_search_fast_mode_user_controls(mock_valyu):
    """Test fast_mode behavior when user sets vs doesn't set it."""
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result
    mock_result = Mock()
    mock_result.content = "Fast mode test"
    mock_result.title = "Fast test"
    mock_result.url = "https://fast-test.com"
    mock_result.source = "fast_source"
    mock_result.price = 0.5
    mock_result.length = 50
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.7

    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    # Test 1: User sets fast_mode=True (model cannot override)
    tool_fixed = ValyuToolSpec(api_key="test_key", fast_mode=True)
    tool_fixed.search(query="test", fast_mode=False)  # Model tries to override

    # Should use user's setting (True), ignoring model's False
    mock_client.search.assert_called_with(
        query="test",
        search_type="all",
        max_num_results=5,
        relevance_threshold=0.5,
        max_price=100,
        start_date=None,
        end_date=None,
        included_sources=None,
        excluded_sources=None,
        response_length=None,
        country_code=None,
        fast_mode=True,  # User's setting used
    )

    mock_client.reset_mock()

    # Test 2: User sets fast_mode=None (model can decide)
    tool_flexible = ValyuToolSpec(api_key="test_key", fast_mode=None)
    tool_flexible.search(query="test", fast_mode=False)  # Model decides

    # Should use model's setting
    mock_client.search.assert_called_with(
        query="test",
        search_type="all",
        max_num_results=5,
        relevance_threshold=0.5,
        max_price=100,
        start_date=None,
        end_date=None,
        included_sources=None,
        excluded_sources=None,
        response_length=None,
        country_code=None,
        fast_mode=False,  # Model's setting used
    )


@patch("valyu.Valyu")
@patch("builtins.print")
def test_search_with_verbose(mock_print, mock_valyu):
    """Test search method with verbose=True."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result object
    mock_result = Mock()
    mock_result.content = "Verbose test content"
    mock_result.title = "Verbose test title"
    mock_result.url = "https://verbose-test.com"
    mock_result.source = "verbose_source"
    mock_result.price = 2.0
    mock_result.length = 200
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.9

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key", verbose=True)

    # Test search with verbose output
    documents = tool.search(query="verbose test query")

    # Verify verbose print was called
    mock_print.assert_called_once_with(f"[Valyu Tool] Response: {mock_response}")

    # Verify document creation
    assert len(documents) == 1
    assert documents[0].text == "Verbose test content"


@patch("valyu.Valyu")
def test_search_multiple_results(mock_valyu):
    """Test search method with multiple results."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create multiple mock result objects
    mock_result1 = Mock()
    mock_result1.content = "First result content"
    mock_result1.title = "First title"
    mock_result1.url = "https://first.com"
    mock_result1.source = "first_source"
    mock_result1.price = 1.0
    mock_result1.length = 100
    mock_result1.data_type = "text"
    mock_result1.relevance_score = 0.9

    mock_result2 = Mock()
    mock_result2.content = "Second result content"
    mock_result2.title = "Second title"
    mock_result2.url = "https://second.com"
    mock_result2.source = "second_source"
    mock_result2.price = 2.0
    mock_result2.length = 200
    mock_result2.data_type = "text"
    mock_result2.relevance_score = 0.8

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result1, mock_result2]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key")

    # Test search with multiple results
    documents = tool.search(query="multi result query")

    # Verify multiple documents were created
    assert len(documents) == 2
    assert documents[0].text == "First result content"
    assert documents[1].text == "Second result content"
    assert documents[0].metadata["title"] == "First title"
    assert documents[1].metadata["title"] == "Second title"


# ========================= Contents API Tests =========================


@patch("valyu.Valyu")
def test_get_contents_basic(mock_valyu):
    """Test basic get_contents functionality."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock contents result object
    mock_result = Mock()
    mock_result.url = "https://example.com"
    mock_result.title = "Example Page"
    mock_result.content = "This is the extracted content from the page."
    mock_result.source = "example.com"
    mock_result.length = 45
    mock_result.data_type = "text"
    mock_result.citation = "Example Page. Retrieved from https://example.com"
    mock_result.summary = None
    mock_result.summary_success = None
    mock_result.image_url = None

    # Mock contents response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.contents.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key")

    # Test get_contents
    documents = tool.get_contents(urls=["https://example.com"])

    # Verify the client was called with correct parameters
    mock_client.contents.assert_called_once_with(
        urls=["https://example.com"],
        summary=None,  # Default
        extract_effort="normal",  # Default
        response_length="short",  # Default
    )

    # Verify document creation
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].text == "This is the extracted content from the page."
    assert documents[0].metadata["url"] == "https://example.com"
    assert documents[0].metadata["title"] == "Example Page"
    assert documents[0].metadata["source"] == "example.com"
    assert documents[0].metadata["length"] == 45
    assert documents[0].metadata["data_type"] == "text"
    assert (
        documents[0].metadata["citation"]
        == "Example Page. Retrieved from https://example.com"
    )


@patch("valyu.Valyu")
def test_get_contents_with_summary(mock_valyu):
    """Test get_contents with summary functionality."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock contents result object with summary
    mock_result = Mock()
    mock_result.url = "https://news-example.com"
    mock_result.title = "News Article"
    mock_result.content = "Full news article content here..."
    mock_result.source = "news-example.com"
    mock_result.length = 500
    mock_result.data_type = "text"
    mock_result.citation = "News Article. Retrieved from https://news-example.com"
    mock_result.summary = "This article discusses recent developments in technology."
    mock_result.summary_success = True
    mock_result.image_url = {"thumbnail": "https://example.com/thumb.jpg"}

    # Mock contents response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.contents.return_value = mock_response

    tool = ValyuToolSpec(
        api_key="test_key",
        contents_summary=True,
        contents_extract_effort="high",
        contents_response_length="medium",
    )

    # Test get_contents
    documents = tool.get_contents(urls=["https://news-example.com"])

    # Verify the client was called with correct parameters
    mock_client.contents.assert_called_once_with(
        urls=["https://news-example.com"],
        summary=True,
        extract_effort="high",
        response_length="medium",
    )

    # Verify document creation with summary
    assert len(documents) == 1
    assert documents[0].text == "Full news article content here..."
    assert documents[0].metadata["url"] == "https://news-example.com"
    assert (
        documents[0].metadata["summary"]
        == "This article discusses recent developments in technology."
    )
    assert documents[0].metadata["summary_success"] is True
    assert documents[0].metadata["image_url"] == {
        "thumbnail": "https://example.com/thumb.jpg"
    }


@patch("valyu.Valyu")
def test_get_contents_multiple_urls(mock_valyu):
    """Test get_contents with multiple URLs."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create multiple mock results
    mock_result1 = Mock()
    mock_result1.url = "https://site1.com"
    mock_result1.title = "Site 1"
    mock_result1.content = "Content from site 1"
    mock_result1.source = "site1.com"
    mock_result1.length = 20
    mock_result1.data_type = "text"
    mock_result1.citation = "Site 1. Retrieved from https://site1.com"
    mock_result1.summary = None
    mock_result1.summary_success = None
    mock_result1.image_url = None

    mock_result2 = Mock()
    mock_result2.url = "https://site2.com"
    mock_result2.title = "Site 2"
    mock_result2.content = "Content from site 2"
    mock_result2.source = "site2.com"
    mock_result2.length = 20
    mock_result2.data_type = "text"
    mock_result2.citation = "Site 2. Retrieved from https://site2.com"
    mock_result2.summary = None
    mock_result2.summary_success = None
    mock_result2.image_url = None

    # Mock contents response object
    mock_response = Mock()
    mock_response.results = [mock_result1, mock_result2]
    mock_client.contents.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key")

    # Test get_contents with multiple URLs
    documents = tool.get_contents(urls=["https://site1.com", "https://site2.com"])

    # Verify the client was called with multiple URLs
    mock_client.contents.assert_called_once_with(
        urls=["https://site1.com", "https://site2.com"],
        summary=None,
        extract_effort="normal",
        response_length="short",
    )

    # Verify multiple documents were created
    assert len(documents) == 2
    assert documents[0].text == "Content from site 1"
    assert documents[1].text == "Content from site 2"
    assert documents[0].metadata["url"] == "https://site1.com"
    assert documents[1].metadata["url"] == "https://site2.com"


@patch("valyu.Valyu")
def test_get_contents_empty_response(mock_valyu):
    """Test get_contents with empty response."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Mock empty contents response
    mock_response = Mock()
    mock_response.results = []
    mock_client.contents.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key")

    # Test get_contents with empty response
    documents = tool.get_contents(urls=["https://failed-site.com"])

    # Verify empty list is returned
    assert len(documents) == 0


@patch("valyu.Valyu")
def test_get_contents_no_response(mock_valyu):
    """Test get_contents when API returns None."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Mock None response
    mock_client.contents.return_value = None

    tool = ValyuToolSpec(api_key="test_key")

    # Test get_contents with None response
    documents = tool.get_contents(urls=["https://failed-site.com"])

    # Verify empty list is returned
    assert len(documents) == 0


# ========================= Retriever Tests =========================


@patch("valyu.Valyu")
def test_valyu_retriever_basic(mock_valyu):
    """Test basic ValyuRetriever functionality."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock contents result object
    mock_result = Mock()
    mock_result.url = "https://example.com"
    mock_result.title = "Example Page"
    mock_result.content = "Retrieved content from example.com"
    mock_result.source = "example.com"
    mock_result.length = 35
    mock_result.data_type = "text"
    mock_result.citation = "Example Page. Retrieved from https://example.com"
    mock_result.summary = None
    mock_result.summary_success = None
    mock_result.image_url = None

    # Mock contents response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.contents.return_value = mock_response

    retriever = ValyuRetriever(api_key="test_key")

    # Create mock query bundle
    mock_query_bundle = Mock()
    mock_query_bundle.query_str = "https://example.com"

    # Test retrieve
    nodes = retriever._retrieve(mock_query_bundle)

    # Verify the client was called with correct parameters
    mock_client.contents.assert_called_once_with(
        urls=["https://example.com"],
        summary=None,
        extract_effort="normal",
        response_length="short",
    )

    # Verify node creation
    assert len(nodes) == 1
    assert isinstance(nodes[0], NodeWithScore)
    assert isinstance(nodes[0].node, TextNode)
    assert nodes[0].score == 1.0
    assert nodes[0].node.text == "Retrieved content from example.com"
    assert nodes[0].node.metadata["url"] == "https://example.com"
    assert nodes[0].node.metadata["title"] == "Example Page"


@patch("valyu.Valyu")
def test_valyu_retriever_multiple_urls(mock_valyu):
    """Test ValyuRetriever with multiple URLs in query."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create multiple mock results
    mock_result1 = Mock()
    mock_result1.url = "https://site1.com"
    mock_result1.title = "Site 1"
    mock_result1.content = "Content from site 1"
    mock_result1.source = "site1.com"
    mock_result1.length = 20
    mock_result1.data_type = "text"
    mock_result1.citation = "Site 1"
    mock_result1.summary = None
    mock_result1.summary_success = None
    mock_result1.image_url = None

    mock_result2 = Mock()
    mock_result2.url = "https://site2.com"
    mock_result2.title = "Site 2"
    mock_result2.content = "Content from site 2"
    mock_result2.source = "site2.com"
    mock_result2.length = 20
    mock_result2.data_type = "text"
    mock_result2.citation = "Site 2"
    mock_result2.summary = None
    mock_result2.summary_success = None
    mock_result2.image_url = None

    # Mock contents response object
    mock_response = Mock()
    mock_response.results = [mock_result1, mock_result2]
    mock_client.contents.return_value = mock_response

    retriever = ValyuRetriever(api_key="test_key")

    # Create mock query bundle with multiple URLs
    mock_query_bundle = Mock()
    mock_query_bundle.query_str = "https://site1.com, https://site2.com"

    # Test retrieve
    nodes = retriever._retrieve(mock_query_bundle)

    # Verify the client was called with multiple URLs
    mock_client.contents.assert_called_once_with(
        urls=["https://site1.com", "https://site2.com"],
        summary=None,
        extract_effort="normal",
        response_length="short",
    )

    # Verify multiple nodes were created
    assert len(nodes) == 2
    assert nodes[0].node.text == "Content from site 1"
    assert nodes[1].node.text == "Content from site 2"


def test_valyu_retriever_url_parsing():
    """Test URL parsing functionality."""
    retriever = ValyuRetriever(api_key="test_key")

    # Test single URL
    urls = retriever._parse_urls_from_query("https://example.com")
    assert urls == ["https://example.com"]

    # Test multiple URLs with comma separation
    urls = retriever._parse_urls_from_query("https://site1.com, https://site2.com")
    assert urls == ["https://site1.com", "https://site2.com"]

    # Test multiple URLs with space separation
    urls = retriever._parse_urls_from_query("https://site1.com https://site2.com")
    assert urls == ["https://site1.com", "https://site2.com"]

    # Test mixed with non-URLs
    urls = retriever._parse_urls_from_query(
        "Please fetch https://example.com and also check https://test.com"
    )
    assert urls == ["https://example.com", "https://test.com"]

    # Test no URLs
    urls = retriever._parse_urls_from_query("This is just text without URLs")
    assert urls == []

    # Test URL limit (should take first 10)
    many_urls = " ".join([f"https://site{i}.com" for i in range(15)])
    urls = retriever._parse_urls_from_query(many_urls)
    assert len(urls) == 10
    assert urls[0] == "https://site0.com"
    assert urls[9] == "https://site9.com"


@patch("valyu.Valyu")
def test_valyu_retriever_empty_query(mock_valyu):
    """Test ValyuRetriever with query containing no URLs."""
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    retriever = ValyuRetriever(api_key="test_key")

    # Create mock query bundle with no URLs
    mock_query_bundle = Mock()
    mock_query_bundle.query_str = "This query has no URLs in it"

    # Test retrieve
    nodes = retriever._retrieve(mock_query_bundle)

    # Verify no API call was made and empty list returned
    mock_client.contents.assert_not_called()
    assert len(nodes) == 0
