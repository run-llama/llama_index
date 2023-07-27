from typing import Callable, Union, List, Type
from abc import ABC, abstractmethod
import asyncio
from datetime import timedelta
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

TimeUnitType = Union[int, float, timedelta]


class RetryStrategy(ABC):
    """Abstract class for retry strategies."""

    @abstractmethod
    def decorate(self, func: Callable):
        """Given a function, return a function that wraps the given function with retry logic.

        Usage:
            @retry_strategy_instance.decorate
            def func_to_retry():
                ...
        """


class NoRetryStrategy(RetryStrategy):
    """Retry strategy that does not retry."""

    def decorate(self, func: Callable):
        return func


class ExponentialBackoffRetryStrategy(RetryStrategy):
    """
    Retry strategy that retries a function with exponential backoff when it raises an exception.
    Uses tenacity's wait_exponential function.
    """

    def __init__(
        self,
        num_attempts: int = 3,
        min_wait: TimeUnitType = 1,
        max_wait: TimeUnitType = 10,
        wait_multiplier: Union[int, float] = 1,
        reraise: bool = True,
        retry_on_exceptions: List[Type[Exception]] = [Exception],
    ):
        if num_attempts < 1:
            raise ValueError(
                f"num_attempts must be greater than or equal to 1. num_attempts={num_attempts}"
            )
        self._num_attempts = num_attempts
        if min_wait > max_wait:
            raise ValueError(
                f"min_wait must be less than or equal to max_wait. min_wait={min_wait}, max_wait={max_wait}"
            )
        self._min_wait = min_wait
        self._max_wait = max_wait
        self._wait_multiplier = wait_multiplier
        self._reraise = reraise
        self._retry_on_exceptions = retry_on_exceptions

    def decorate(self, func: Callable):
        retry_condition = retry_if_exception_type(*self._retry_on_exceptions)

        if asyncio.iscoroutinefunction(func):

            @retry(
                stop=stop_after_attempt(self._num_attempts),
                wait=wait_exponential(
                    multiplier=self._wait_multiplier,
                    min=self._min_wait,
                    max=self._max_wait,
                ),
                reraise=self._reraise,
                retry=retry_condition,
            )
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return async_wrapper

        @retry(
            stop=stop_after_attempt(self._num_attempts),
            wait=wait_exponential(
                multiplier=self._wait_multiplier, min=self._min_wait, max=self._max_wait
            ),
            reraise=self._reraise,
            retry=retry_condition,
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
