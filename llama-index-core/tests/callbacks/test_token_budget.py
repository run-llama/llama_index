import pytest
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.callbacks.token_counting import TokenBudgetHandler


def test_budget_enforcement():
    """Test that the budget handler raises an error when limit is hit."""
    # 1. Setup: Create a handler with a strict budget of 100 tokens
    handler = TokenBudgetHandler(token_budget=100, verbose=True)

    # 2. Simulate a safe event (Manually setting count to 50)
    # We cheat and set the internal counter directly to simulate usage
    handler.llm_token_counts = [type("MockEvent", (), {"total_token_count": 50})]

    # This should pass (50 < 100)
    handler.on_event_start(event_type=CBEventType.LLM)

    # 3. Simulate an unsafe event (Manually setting count to 150)
    handler.llm_token_counts = [type("MockEvent", (), {"total_token_count": 150})]

    # This should CRASH (150 > 100)
    with pytest.raises(ValueError) as excinfo:
        handler.on_event_start(event_type=CBEventType.LLM)

    # 4. Verify the error message is correct
    assert "Token budget exceeded" in str(excinfo.value)
    assert "Limit: 100" in str(excinfo.value)
