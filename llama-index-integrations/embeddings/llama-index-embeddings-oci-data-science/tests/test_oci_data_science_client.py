from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from llama_index.embeddings.oci_data_science.client import (
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

    def test_init_invalid_auth(self):
        """Ensures that ValueError is raised when auth signer is invalid."""
        with pytest.raises(ValueError):
            BaseClient(endpoint=self.endpoint, auth={"signer": None})

    def test_prepare_headers(self):
        """Ensures that headers are prepared correctly."""
        headers = {"Custom-Header": "Value"}
        result = self.base_client._prepare_headers(headers=headers)
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Custom-Header": "Value",
        }
        assert result == expected_headers


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

    def test_auth_not_provided(self):
        """Ensures that error will be thrown what auth signer not provided."""
        with pytest.raises(ImportError):
            Client(
                endpoint=self.endpoint,
                retries=self.retries,
                backoff_factor=self.backoff_factor,
                timeout=self.timeout,
            )

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

    def test_embeddings_request(self):
        """Ensures that embeddings method calls _request when stream=False."""
        response_json = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [
                        -0.025584878,
                        0.023328023,
                        -0.03014998,
                    ],
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [
                        -0.025584878,
                        0.023328023,
                        -0.03014998,
                    ],
                },
            ],
        }
        response_mock = Mock()
        response_mock.json.return_value = response_json
        response_mock.status_code = 200

        self.client._client.post.return_value = response_mock

        result = self.client.embeddings(
            input=["Hello", "World"], payload={"param1": "value1"}
        )

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


@pytest.mark.asyncio()
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

    async def test_generate_request(self):
        """Ensures that generate method calls _request when stream=False."""
        response_json = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [
                        -0.025584878,
                        0.023328023,
                        -0.03014998,
                    ],
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [
                        -0.025584878,
                        0.023328023,
                        -0.03014998,
                    ],
                },
            ],
        }
        response_mock = AsyncMock()
        response_mock.status_code = 200
        response_mock.json = AsyncMock(return_value=response_json)
        response_mock.raise_for_status = Mock()

        self.client._client.post.return_value = response_mock

        result = await self.client.embeddings(
            input=["Hello", "World"], payload={"param1": "value1"}
        )

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
