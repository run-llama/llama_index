from typing_extensions import Literal

from httpx import Request, Response


class APIError(Exception):
    message: str
    request: Request

    def __init__(self, message: str, request: Request) -> None:
        super().__init__(message)
        self.request = request
        self.message = message


class APIResponseValidationError(APIError):
    response: Response
    status_code: int

    def __init__(self, request: Request, response: Response) -> None:
        super().__init__("Data returned by API invalid for expected schema.", request)
        self.response = response
        self.status_code = response.status_code


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: Response
    status_code: int

    body: object
    """The API response body.

    If the API responded with a valid JSON structure then this property will be the 
    decoded result.
    If it isn't a valid JSON structure then this will be the raw response.
    """

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request)
        self.response = response
        self.status_code = response.status_code
        self.body = body


class BadRequestError(APIStatusError):
    status_code: Literal[400]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 400


class AuthenticationError(APIStatusError):
    status_code: Literal[401]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 401


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 403


class NotFoundError(APIStatusError):
    status_code: Literal[404]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 404


class ConflictError(APIStatusError):
    status_code: Literal[409]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 409


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 422


class RateLimitError(APIStatusError):
    status_code: Literal[429]

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = 429


class InternalServerError(APIStatusError):
    status_code: int

    def __init__(
        self, message: str, *, request: Request, response: Response, body: object
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = response.status_code


class APIConnectionError(APIError):
    def __init__(self, request: Request, message: str = "Connection error.") -> None:
        super().__init__(message, request)


class APITimeoutError(APIConnectionError):
    def __init__(self, request: Request) -> None:
        super().__init__(request, "Request timed out.")
