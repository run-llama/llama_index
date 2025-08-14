# Retry steps execution

A step that fails its execution might result in the failure of the entire workflow, but oftentimes errors are
expected and the execution can be safely retried. Think of a HTTP request that times out because of a transient
congestion of the network, or an external API call that hits a rate limiter.

For all those situation where you want the step to try again, you can use a "Retry Policy". A retry policy is an object
that instructs the workflow to execute a step multiple times, dictating how much time has to pass before a new attempt.
Policies take into consideration how much time passed since the first failure, how many consecutive failures happened
and which was the last error occurred.

To set a policy for a specific step, all you have to do is passing a policy object to the `@step` decorator:


```python
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy


class MyWorkflow(Workflow):
    # ...more workflow definition...

    # This policy will retry this step on failure every 5 seconds for at most 10 times
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=10))
    async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        result = flaky_call()  # this might raise
        return StopEvent(result=result)
```

You can see the [API docs](../../api_reference/workflow/retry_policy.md) for a detailed description of the policies
available in the framework. If you can't find a policy that's suitable for your use case, you can easily write a
custom one. The only requirement for custom policies is to write a Python class that respects the `RetryPolicy`
protocol. In other words, your custom policy class must have a method with the following signature:

```python
def next(
    self, elapsed_time: float, attempts: int, error: Exception
) -> Optional[float]:
    ...
```

For example, this is a retry policy that's excited about the weekend and only retries a step if it's Friday:

```python
from datetime import datetime


class RetryOnFridayPolicy:
    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> Optional[float]:
        if datetime.today().strftime("%A") == "Friday":
            # retry in 5 seconds
            return 5
        # tell the workflow we don't want to retry
        return None
```
