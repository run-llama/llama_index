import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from llama_index.llms.oci_data_science.client import (
    AsyncClient,
    BaseClient,
    Client,
    ExtendedRequestException,
    OCIAuth,
    _create_retry_decorator,
    _retry_decorator,
    _should_retry_exception,
)


class TestOCIAuth:
    """Unit tests for OCIAuth class."""

    def setup_method(self):
        self.signer_mock = Mock()
        self.oci_auth = OCIAuth(self.signer_mock)

    def test_auth_flow(self):
        """Ensures that the auth_flow signs the request correctly."""
        request = httpx.Request("POST", "https://example.com")
        prepared_request_mock = Mock()
        prepared_request_mock.headers = {"Authorization": "Signed"}
        with patch("requests.Request") as mock_requests_request:
            mock_requests_request.return_value = Mock()
            mock_requests_request.return_value.prepare.return_value = (
                prepared_request_mock
            )
            self.signer_mock.do_request_sign = Mock()

            list(self.oci_auth.auth_flow(request))

            self.signer_mock.do_request_sign.assert_called()
            assert request.headers.get("Authorization") == "Signed"


class TestExtendedRequestException:
    """Unit tests for ExtendedRequestException."""

    def test_exception_attributes(self):
        """Ensures the exception stores the correct attributes."""
        original_exception = Exception("Original error")
        response_text = "Error response text"
        message = "Extended error message"

        exception = ExtendedRequestException(message, original_exception, response_text)

        assert str(exception) == message
        assert exception.original_exception == original_exception
        assert exception.response_text == response_text


class TestShouldRetryException:
    """Unit tests for _should_retry_exception function."""

    def test_http_status_error_in_force_list(self):
        """Ensures it returns True for HTTPStatusError with status in STATUS_FORCE_LIST."""
        response_mock = Mock()
        response_mock.status_code = 500
        original_exception = httpx.HTTPStatusError(
            "Error", request=None, response=response_mock
        )
        exception = ExtendedRequestException(
            "Message", original_exception, "Response text"
        )

        result = _should_retry_exception(exception)
        assert result is True

    def test_http_status_error_not_in_force_list(self):
        """Ensures it returns False for HTTPStatusError with status not in STATUS_FORCE_LIST."""
        response_mock = Mock()
        response_mock.status_code = 404
        original_exception = httpx.HTTPStatusError(
            "Error", request=None, response=response_mock
        )
        exception = ExtendedRequestException(
            "Message", original_exception, "Response text"
        )

        result = _should_retry_exception(exception)
        assert result is False

    def test_http_request_error(self):
        """Ensures it returns True for RequestError."""
        original_exception = httpx.RequestError("Error")
        exception = ExtendedRequestException(
            "Message", original_exception, "Response text"
        )

        result = _should_retry_exception(exception)
        assert result is True

    def test_other_exception(self):
        """Ensures it returns False for other exceptions."""
        original_exception = Exception("Some other error")
        exception = ExtendedRequestException(
            "Message", original_exception, "Response text"
        )

        result = _should_retry_exception(exception)
        assert result is False


class TestCreateRetryDecorator:
    """Unit tests for _create_retry_decorator function."""

    def test_create_retry_decorator(self):
        """Ensures the retry decorator is created with correct parameters."""
        max_retries = 5
        backoff_factor = 2
        random_exponential = False
        stop_after_delay_seconds = 100
        min_seconds = 1
        max_seconds = 10

        retry_decorator = _create_retry_decorator(
            max_retries,
            backoff_factor,
            random_exponential,
            stop_after_delay_seconds,
            min_seconds,
            max_seconds,
        )

        assert callable(retry_decorator)


class TestRetryDecorator:
    """Unit tests for _retry_decorator function."""

    def test_retry_decorator_no_retries(self):
        """Ensures the function is called directly when retries is 0."""

        class TestClass:
            retries = 0
            backoff_factor = 1
            timeout = 10

            @_retry_decorator
            def test_method(self):
                return "Success"

        test_instance = TestClass()
        result = test_instance.test_method()
        assert result == "Success"

    def test_retry_decorator_with_retries(self):
        """Ensures the function retries upon exception."""

        class TestClass:
            retries = 3
            backoff_factor = 0.1
            timeout = 10

            call_count = 0

            @_retry_decorator
            def test_method(self):
                self.call_count += 1
                if self.call_count < 3:
                    raise ExtendedRequestException(
                        "Error",
                        original_exception=httpx.RequestError("Error"),
                        response_text="test",
                    )
                return "Success"

        test_instance = TestClass()
        result = test_instance.test_method()
        assert result == "Success"
        assert test_instance.call_count == 3

    def test_retry_decorator_exceeds_retries(self):
        """Ensures the function raises exception after exceeding retries."""

        class TestClass:
            retries = 3
            backoff_factor = 0.1
            timeout = 10

            call_count = 0

            @_retry_decorator
            def test_method(self):
                self.call_count += 1
                raise ExtendedRequestException(
                    "Error",
                    original_exception=httpx.RequestError("Error"),
                    response_text="test",
                )

        test_instance = TestClass()
        with pytest.raises(ExtendedRequestException):
            test_instance.test_method()
        assert test_instance.call_count == 3  # initial attempt + 2 retries


class TestBaseClient:
    """Unit tests for BaseClient class."""

    def setup_method(self):
        self.endpoint = "https://example.com/api"
        self.auth_mock = {"signer": Mock()}
        self.retries = 3
        self.backoff_factor = 2
        self.timeout = 30

        self.base_client = BaseClient(
            endpoint=self.endpoint,
            auth=self.auth_mock,
            retries=self.retries,
            backoff_factor=self.backoff_factor,
            timeout=self.timeout,
        )

    def test_init(self):
        """Ensures that the client is initialized correctly."""
        assert self.base_client.endpoint == self.endpoint
        assert self.base_client.retries == self.retries
        assert self.base_client.backoff_factor == self.backoff_factor
        assert self.base_client.timeout == self.timeout
        assert isinstance(self.base_client.auth, OCIAuth)

    # def test_init_default_auth(self):
    #     """Ensures that default auth is used when auth is None."""
    #     with patch.object(authutil, "default_signer", return_value=self.auth_mock):
    #         client = BaseClient(endpoint=self.endpoint)
    #         assert client.auth is not None

    def test_auth_not_provided(self):
        """Ensures that error will be thrown what auth signer not provided."""
        with pytest.raises(ImportError):
            Client(
                endpoint=self.endpoint,
                retries=self.retries,
                backoff_factor=self.backoff_factor,
                timeout=self.timeout,
            )

    def test_init_invalid_auth(self):
        """Ensures that ValueError is raised when auth signer is invalid."""
        with pytest.raises(ValueError):
            BaseClient(endpoint=self.endpoint, auth={"signer": None})

    def test_prepare_headers(self):
        """Ensures that headers are prepared correctly."""
        headers = {"Custom-Header": "Value"}
        result = self.base_client._prepare_headers(stream=False, headers=headers)
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Custom-Header": "Value",
        }
        assert result == expected_headers

    def test_prepare_headers_stream(self):
        """Ensures that headers are prepared correctly for streaming."""
        headers = {"Custom-Header": "Value"}
        result = self.base_client._prepare_headers(stream=True, headers=headers)
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "enable-streaming": "true",
            "Custom-Header": "Value",
        }
        assert result == expected_headers

    def test_parse_streaming_line_valid(self):
        """Ensures that a valid streaming line is parsed correctly."""
        line = 'data: {"key": "value"}'
        result = self.base_client._parse_streaming_line(line)
        assert result == {"key": "value"}

    def test_parse_streaming_line_invalid_json(self):
        """Ensures that JSONDecodeError is raised for invalid JSON."""
        line = "data: invalid json"
        with pytest.raises(json.JSONDecodeError):
            self.base_client._parse_streaming_line(line)

    def test_parse_streaming_line_empty(self):
        """Ensures that None is returned for empty or end-of-stream lines."""
        line = ""
        result = self.base_client._parse_streaming_line(line)
        assert result is None

        line = "[DONE]"
        result = self.base_client._parse_streaming_line(line)
        assert result is None

    def test_parse_streaming_line_error_object(self):
        """Ensures that an exception is raised for error objects in the stream."""
        line = 'data: {"object": "error", "message": "Error message"}'
        with pytest.raises(Exception) as exc_info:
            self.base_client._parse_streaming_line(line)
        assert "Error in streaming response: Error message" in str(exc_info.value)


class TestClient:
    """Unit tests for Client class."""

    def setup_method(self):
        self.endpoint = "https://example.com/api"
        self.auth_mock = {"signer": Mock()}
        self.retries = 2
        self.backoff_factor = 0.1
        self.timeout = 10

        self.client = Client(
            endpoint=self.endpoint,
            auth=self.auth_mock,
            retries=self.retries,
            backoff_factor=self.backoff_factor,
            timeout=self.timeout,
        )
        # Mock the internal HTTPX client
        self.client._client = Mock()

    def test_request_success(self):
        """Ensures that _request returns JSON response on success."""
        payload = {"prompt": "Hello"}
        response_json = {"choices": [{"text": "Hi"}]}
        response_mock = Mock()
        response_mock.json.return_value = response_json
        response_mock.status_code = 200

        self.client._client.post.return_value = response_mock

        result = self.client._request(payload)

        assert result == response_json

    def test_request_http_error(self):
        """Ensures that _request raises ExtendedRequestException on HTTP error."""
        payload = {"prompt": "Hello"}
        response_mock = Mock()
        response_mock.status_code = 500
        response_mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=None, response=response_mock
        )
        response_mock.text = "Internal Server Error"

        self.client._client.post.return_value = response_mock

        with pytest.raises(ExtendedRequestException) as exc_info:
            self.client._request(payload)

        assert "Request failed" in str(exc_info.value)
        assert exc_info.value.response_text == "Internal Server Error"

    def test_stream_success(self):
        """Ensures that _stream yields parsed lines on success."""
        payload = {"prompt": "Hello"}
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.iter_lines.return_value = [
            b'data: {"key": "value1"}',
            b'data: {"key": "value2"}',
            b"[DONE]",
        ]
        # Mock the context manager
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = response_mock
        self.client._client.stream.return_value = stream_cm

        result = list(self.client._stream(payload))

        assert result == [{"key": "value1"}, {"key": "value2"}]

    @patch("time.sleep", return_value=None)
    def test_stream_retry_on_exception(self, mock_sleep):
        """Ensures that _stream retries on exceptions and raises after retries exhausted."""
        payload = {"prompt": "Hello"}

        # Mock the exception to be raised
        def side_effect(*args, **kwargs):
            raise httpx.RequestError("Connection error")

        # Mock the context manager
        self.client._client.stream.side_effect = side_effect

        with pytest.raises(ExtendedRequestException):
            list(self.client._stream(payload))

        assert (
            self.client._client.stream.call_count == self.retries + 1
        )  # initial attempt + retries

    def test_generate_stream(self):
        """Ensures that generate method calls _stream when stream=True."""
        payload = {"prompt": "Hello"}
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.iter_lines.return_value = [b'data: {"key": "value"}', b"[DONE]"]
        # Mock the context manager
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = response_mock
        self.client._client.stream.return_value = stream_cm

        result = list(self.client.generate(prompt="Hello", stream=True))

        assert result == [{"key": "value"}]

    def test_generate_request(self):
        """Ensures that generate method calls _request when stream=False."""
        payload = {"prompt": "Hello"}
        response_json = {"choices": [{"text": "Hi"}]}
        response_mock = Mock()
        response_mock.json.return_value = response_json
        response_mock.status_code = 200

        self.client._client.post.return_value = response_mock

        result = self.client.generate(prompt="Hello", stream=False)

        assert result == response_json

    def test_chat_stream(self):
        """Ensures that chat method calls _stream when stream=True."""
        messages = [{"role": "user", "content": "Hello"}]
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.iter_lines.return_value = [b'data: {"key": "value"}', b"[DONE]"]
        # Mock the context manager
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = response_mock
        self.client._client.stream.return_value = stream_cm

        result = list(self.client.chat(messages=messages, stream=True))

        assert result == [{"key": "value"}]

    def test_chat_request(self):
        """Ensures that chat method calls _request when stream=False."""
        messages = [{"role": "user", "content": "Hello"}]
        response_json = {"choices": [{"message": {"content": "Hi"}}]}
        response_mock = Mock()
        response_mock.json.return_value = response_json
        response_mock.status_code = 200

        self.client._client.post.return_value = response_mock

        result = self.client.chat(messages=messages, stream=False)

        assert result == response_json

    def test_close(self):
        """Ensures that close method closes the client."""
        self.client._client.close = Mock()
        self.client.close()
        self.client._client.close.assert_called_once()

    def test_is_closed(self):
        """Ensures that is_closed returns the client's is_closed status."""
        self.client._client.is_closed = False
        assert not self.client.is_closed()
        self.client._client.is_closed = True
        assert self.client.is_closed()

    def test_context_manager(self):
        """Ensures that the client can be used as a context manager."""
        self.client.close = Mock()
        with self.client as client_instance:
            assert client_instance == self.client
        self.client.close.assert_called_once()

    def test_del(self):
        """Ensures that __del__ method closes the client."""
        client = Client(
            endpoint=self.endpoint,
            auth=self.auth_mock,
            retries=self.retries,
            backoff_factor=self.backoff_factor,
            timeout=self.timeout,
        )
        client.close = Mock()
        client.__del__()  # Manually invoke __del__
        client.close.assert_called_once()


@pytest.mark.asyncio
class TestAsyncClient:
    """Unit tests for AsyncClient class."""

    def setup_method(self):
        self.endpoint = "https://example.com/api"
        self.auth_mock = {"signer": Mock()}
        self.retries = 2
        self.backoff_factor = 0.1
        self.timeout = 10

        self.client = AsyncClient(
            endpoint=self.endpoint,
            auth=self.auth_mock,
            retries=self.retries,
            backoff_factor=self.backoff_factor,
            timeout=self.timeout,
        )
        # Mock the internal HTTPX client
        self.client._client = AsyncMock()
        self.client._client.is_closed = False

    def async_iter(self, items):
        """Helper function to create an async iterator from a list."""

        async def generator():
            for item in items:
                yield item

        return generator()

    async def test_request_success(self):
        """Ensures that _request returns JSON response on success."""
        payload = {"prompt": "Hello"}
        response_json = {"choices": [{"text": "Hi"}]}
        response_mock = AsyncMock()
        response_mock.status_code = 200
        response_mock.json = AsyncMock(return_value=response_json)
        response_mock.raise_for_status = Mock()
        self.client._client.post.return_value = response_mock
        result = await self.client._request(payload)
        assert await result == response_json

    async def test_request_http_error(self):
        """Ensures that _request raises ExtendedRequestException on HTTP error."""
        payload = {"prompt": "Hello"}
        response_mock = MagicMock()
        response_mock.status_code = 500
        response_mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=None, response=response_mock
        )
        response_mock.text = "Internal Server Error"

        self.client._client.post.return_value = response_mock

        with pytest.raises(ExtendedRequestException) as exc_info:
            await self.client._request(payload)

        assert "Request failed" in str(exc_info.value)
        assert exc_info.value.response_text == "Internal Server Error"

    async def test_stream_success(self):
        """Ensures that _stream yields parsed lines on success."""
        payload = {"prompt": "Hello"}
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.raise_for_status = Mock()
        response_mock.aiter_lines.return_value = self.async_iter(
            ['data: {"key": "value1"}', 'data: {"key": "value2"}', "[DONE]"]
        )

        # Define an async context manager
        @asynccontextmanager
        async def stream_context_manager(*args, **kwargs):
            yield response_mock

        # Mock the stream method to return our context manager
        self.client._client.stream = Mock(side_effect=stream_context_manager)

        result = []
        async for item in self.client._stream(payload):
            result.append(item)

        assert result == [{"key": "value1"}, {"key": "value2"}]

    @patch("asyncio.sleep", return_value=None)
    async def test_stream_retry_on_exception(self, mock_sleep):
        """Ensures that _stream retries on exceptions and raises after retries exhausted."""
        payload = {"prompt": "Hello"}

        # Define an async context manager that raises an exception
        @asynccontextmanager
        async def stream_context_manager(*args, **kwargs):
            raise httpx.RequestError("Connection error")
            yield  # This is never reached

        # Mock the stream method to use our context manager
        self.client._client.stream = Mock(side_effect=stream_context_manager)

        with pytest.raises(ExtendedRequestException):
            async for _ in self.client._stream(payload):
                pass

        assert (
            self.client._client.stream.call_count == self.retries + 1
        )  # initial attempt + retries

    async def test_generate_stream(self):
        """Ensures that generate method calls _stream when stream=True."""
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.raise_for_status = Mock()
        response_mock.aiter_lines.return_value = self.async_iter(
            ['data: {"key": "value"}', "[DONE]"]
        )

        @asynccontextmanager
        async def stream_context_manager(*args, **kwargs):
            yield response_mock

        self.client._client.stream = Mock(side_effect=stream_context_manager)

        result = []
        async for item in await self.client.generate(prompt="Hello", stream=True):
            result.append(item)

        assert result == [{"key": "value"}]

    async def test_generate_request(self):
        """Ensures that generate method calls _request when stream=False."""
        payload = {"prompt": "Hello"}
        response_json = {"choices": [{"text": "Hi"}]}
        response_mock = AsyncMock()
        response_mock.status_code = 200
        response_mock.json = AsyncMock(return_value=response_json)
        response_mock.raise_for_status = Mock()

        self.client._client.post.return_value = response_mock

        result = await self.client.generate(prompt="Hello", stream=False)

        assert await result == response_json

    async def test_chat_stream(self):
        """Ensures that chat method calls _stream when stream=True."""
        messages = [{"role": "user", "content": "Hello"}]
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.raise_for_status = Mock()
        response_mock.aiter_lines.return_value = self.async_iter(
            ['data: {"key": "value"}', "[DONE]"]
        )

        @asynccontextmanager
        async def stream_context_manager(*args, **kwargs):
            yield response_mock

        self.client._client.stream = Mock(side_effect=stream_context_manager)

        result = []
        async for item in await self.client.chat(messages=messages, stream=True):
            result.append(item)

        assert result == [{"key": "value"}]

    async def test_chat_request(self):
        """Ensures that chat method calls _request when stream=False."""
        messages = [{"role": "user", "content": "Hello"}]
        response_json = {"choices": [{"message": {"content": "Hi"}}]}
        response_mock = AsyncMock()
        response_mock.status_code = 200
        response_mock.json = AsyncMock(return_value=response_json)
        response_mock.raise_for_status = Mock()

        self.client._client.post.return_value = response_mock

        result = await self.client.chat(messages=messages, stream=False)

        assert await result == response_json

    async def test_close(self):
        """Ensures that close method closes the client."""
        self.client._client.aclose = AsyncMock()
        await self.client.close()
        self.client._client.aclose.assert_called_once()

    async def test_is_closed(self):
        """Ensures that is_closed returns the client's is_closed status."""
        self.client._client.is_closed = False
        assert not self.client.is_closed()
        self.client._client.is_closed = True
        assert self.client.is_closed()

    async def test_context_manager(self):
        """Ensures that the client can be used as a context manager."""
        self.client.close = AsyncMock()
        async with self.client as client_instance:
            assert client_instance == self.client
        self.client.close.assert_called_once()

    async def test_del(self):
        """Ensures that __del__ method closes the client."""
        client = AsyncClient(
            endpoint=self.endpoint,
            auth=self.auth_mock,
            retries=self.retries,
            backoff_factor=self.backoff_factor,
            timeout=self.timeout,
        )
        client.close = AsyncMock()
        await client.__aexit__(None, None, None)  # Manually invoke __aexit__
        client.close.assert_called_once()
