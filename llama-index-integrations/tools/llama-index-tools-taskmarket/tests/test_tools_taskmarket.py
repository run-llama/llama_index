from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.taskmarket import TaskmarketToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in TaskmarketToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    spec = TaskmarketToolSpec()
    assert "list_tasks" in spec.spec_functions
    assert "get_task" in spec.spec_functions
    assert "list_submissions" in spec.spec_functions
    assert "create_task" in spec.spec_functions


def test_create_task_refuses_without_authorization():
    spec = TaskmarketToolSpec()
    out = spec.create_task(description="test", reward=1.0, duration_hours=24)
    assert out.startswith("REFUSED")
    assert "Nothing was created" in out


def test_create_task_refuses_wrong_authorization():
    spec = TaskmarketToolSpec()
    out = spec.create_task(
        description="test",
        reward=1.0,
        duration_hours=24,
        authorization="authorize 999 USDC for taskmarket task",
    )
    assert out.startswith("REFUSED")
    assert "must exactly match" in out


def test_create_task_refuses_over_max_spend():
    spec = TaskmarketToolSpec(max_spend=0.5)
    out = spec.create_task(
        description="test",
        reward=5.0,
        duration_hours=24,
        authorization="authorize 5.0 USDC for taskmarket task",
    )
    assert out.startswith("REFUSED")
    assert "exceeds max_spend" in out


def test_create_task_refuses_bad_duration():
    spec = TaskmarketToolSpec()
    out = spec.create_task(
        description="test",
        reward=1.0,
        duration_hours=0,
        authorization="authorize 1.0 USDC for taskmarket task",
    )
    assert out.startswith("REFUSED")
    assert "duration_hours" in out


def test_extract_task_id():
    fake_id = "0x" + "ab" * 32  # 64 hex chars, same shape as real task ids
    assert TaskmarketToolSpec._extract_task_id(f"created task {fake_id} ok") == fake_id
    assert TaskmarketToolSpec._extract_task_id("no id here") is None
