from typing import Protocol, Optional, runtime_checkable


@runtime_checkable
class RetryPolicy(Protocol):
    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> Optional[float]:
        """Decide if we should make another retry, returning the number of seconds to wait before the next run.

        Args:
            elapsed_time: Time in seconds that passed since the last attempt.
            attempts: The number of attempts done so far.
            error: The last error occurred.

        Returns:
            The amount of seconds to wait before the next attempt, or None if we stop retrying.
        """


class ConstantDelayRetryPolicy:
    def __init__(self, maximum_attempts: int = 3, delay: float = 5) -> None:
        self.maximum_attempts = maximum_attempts
        self.delay = delay

    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> Optional[float]:
        if attempts >= self.maximum_attempts:
            return None

        return self.delay
