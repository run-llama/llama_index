"""Tests for HerokuToolSpec."""

import pytest
from pytest_httpx import HTTPXMock

from llama_index.tools.heroku import HerokuToolSpec, create_heroku_tools


class TestHerokuToolSpec:
    """Test suite for HerokuToolSpec class."""

    @pytest.fixture
    def tool_spec(self) -> HerokuToolSpec:
        """Create a tool spec for testing."""
        return HerokuToolSpec(
            api_key="test-api-key",
            app_name="test-app",
        )

    @pytest.fixture
    def mock_sql_response(self) -> dict:
        """Create a mock SQL response."""
        return {
            "output": "id | name\n---+------\n1  | Alice\n2  | Bob",
        }

    @pytest.fixture
    def mock_python_response(self) -> dict:
        """Create a mock Python execution response."""
        return {
            "output": "4950",
        }

    @pytest.fixture
    def mock_js_response(self) -> dict:
        """Create a mock JavaScript execution response."""
        return {
            "output": "6",
        }

    @pytest.fixture
    def mock_app_info_response(self) -> dict:
        """Create a mock app info response."""
        return {
            "output": "App: test-app\nStack: heroku-24\nDynos: 1",
        }

    def test_initialization(self, tool_spec: HerokuToolSpec) -> None:
        """Test tool spec initialization."""
        assert tool_spec.api_key == "test-api-key"
        assert tool_spec.app_name == "test-app"
        assert tool_spec.base_url == "https://us.inference.heroku.com"
        assert tool_spec.timeout == 120.0

    def test_spec_functions(self, tool_spec: HerokuToolSpec) -> None:
        """Test that spec_functions are defined."""
        assert "run_sql" in tool_spec.spec_functions
        assert "run_python" in tool_spec.spec_functions
        assert "run_javascript" in tool_spec.spec_functions
        assert "get_app_info" in tool_spec.spec_functions

    def test_to_tool_list(self, tool_spec: HerokuToolSpec) -> None:
        """Test converting spec to tool list."""
        tools = tool_spec.to_tool_list()

        assert len(tools) == 4
        tool_names = [tool.metadata.name for tool in tools]
        assert "run_sql" in tool_names
        assert "run_python" in tool_names
        assert "run_javascript" in tool_names
        assert "get_app_info" in tool_names

    def test_run_sql(
        self,
        httpx_mock: HTTPXMock,
        tool_spec: HerokuToolSpec,
        mock_sql_response: dict,
    ) -> None:
        """Test SQL execution."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/agents/heroku",
            method="POST",
            json=mock_sql_response,
        )

        result = tool_spec.run_sql("SELECT * FROM users LIMIT 2")

        assert "Alice" in result
        assert "Bob" in result

    def test_run_python(
        self,
        httpx_mock: HTTPXMock,
        tool_spec: HerokuToolSpec,
        mock_python_response: dict,
    ) -> None:
        """Test Python code execution."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/agents/heroku",
            method="POST",
            json=mock_python_response,
        )

        result = tool_spec.run_python("print(sum(range(100)))")

        assert result == "4950"

    def test_run_javascript(
        self,
        httpx_mock: HTTPXMock,
        tool_spec: HerokuToolSpec,
        mock_js_response: dict,
    ) -> None:
        """Test JavaScript code execution."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/agents/heroku",
            method="POST",
            json=mock_js_response,
        )

        result = tool_spec.run_javascript("console.log([1,2,3].reduce((a,b) => a+b))")

        assert result == "6"

    def test_get_app_info(
        self,
        httpx_mock: HTTPXMock,
        tool_spec: HerokuToolSpec,
        mock_app_info_response: dict,
    ) -> None:
        """Test getting app info."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/agents/heroku",
            method="POST",
            json=mock_app_info_response,
        )

        result = tool_spec.get_app_info()

        assert "test-app" in result
        assert "heroku-24" in result

    def test_request_headers(self, tool_spec: HerokuToolSpec) -> None:
        """Test that correct headers are generated."""
        headers = tool_spec._get_headers()

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_error_response_handling(
        self,
        httpx_mock: HTTPXMock,
        tool_spec: HerokuToolSpec,
    ) -> None:
        """Test handling of error responses."""
        httpx_mock.add_response(
            url="https://us.inference.heroku.com/v1/agents/heroku",
            method="POST",
            json={"error": "Database connection failed"},
        )

        result = tool_spec.run_sql("SELECT * FROM users")

        assert "Error:" in result
        assert "Database connection failed" in result

    def test_custom_base_url(self) -> None:
        """Test custom base URL configuration."""
        tool_spec = HerokuToolSpec(
            api_key="test-key",
            app_name="test-app",
            base_url="https://custom.heroku.com",
        )

        assert tool_spec.base_url == "https://custom.heroku.com"

    def test_custom_timeout(self) -> None:
        """Test custom timeout configuration."""
        tool_spec = HerokuToolSpec(
            api_key="test-key",
            app_name="test-app",
            timeout=300.0,
        )

        assert tool_spec.timeout == 300.0


class TestCreateHerokuTools:
    """Test the create_heroku_tools convenience function."""

    def test_create_heroku_tools(self) -> None:
        """Test creating tools via convenience function."""
        tools = create_heroku_tools(
            api_key="test-key",
            app_name="test-app",
        )

        assert len(tools) == 4
        tool_names = [tool.metadata.name for tool in tools]
        assert "run_sql" in tool_names

    def test_create_heroku_tools_with_options(self) -> None:
        """Test creating tools with custom options."""
        tools = create_heroku_tools(
            api_key="test-key",
            app_name="test-app",
            base_url="https://custom.heroku.com",
            timeout=300.0,
        )

        assert len(tools) == 4
