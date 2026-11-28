import base64
import json
import pickle
from dataclasses import dataclass
from typing import Any

import pytest
from llama_index.core.workflow import (
    Context,
    JsonPickleSerializer,
    JsonSerializer,
    PickleSerializer,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


@dataclass
class CustomObject:
    name: str
    count: int


class RememberWorkflow(Workflow):
    @step
    async def remember(self, ctx: Context, ev: StartEvent) -> StopEvent:
        before = await ctx.store.get("cell", default=None)
        await ctx.store.set("cell", ev.text)
        return StopEvent(result=before)


PICKLE_LOOKALIKE_JSON_VALUES: list[Any] = [
    "SwEu",
    "SwIu",
    "Ti4=",
    "iC4=",
    "iS4=",
    "XS4=",
    "fS4=",
    base64.b64encode(pickle.dumps(CustomObject(name="test", count=42))).decode("utf-8"),
    ["SwEu"],
    {"cell": "SwEu"},
    {"cells": ["SwEu", "Ti4="]},
]

ARBITRARY_JSON_VALUES: list[Any] = [
    0,
    1234,
    -42,
    3.14159,
    True,
    False,
    None,
    "",
    "hello world",
    "ünïcödé ✓",
    [],
    {},
    [1, "SwEu", True, None, 3.14],
    {"a": 1, "b": "SwEu", "c": [True, False]},
    {"nested": {"deep": {"value": "SwEu"}}},
]

PICKLE_ONLY_VALUES: list[Any] = [
    CustomObject(name="example", count=99),
    {1, 2, 3},
    b"binary\x00\xffdata",
    {"custom": CustomObject(name="nested", count=1), "set": {4, 5, 6}},
    [b"raw", {10, 20}],
]


def test_json_pickle_serializer_alias() -> None:
    assert JsonPickleSerializer is PickleSerializer


@pytest.mark.parametrize("value", PICKLE_LOOKALIKE_JSON_VALUES)
def test_pickle_serializer_round_trips_pickle_lookalike_json_values(
    value: Any,
) -> None:
    serializer = PickleSerializer()
    serialized = serializer.serialize(value)
    assert serializer.deserialize(serialized) == value


@pytest.mark.parametrize("value", ARBITRARY_JSON_VALUES)
def test_pickle_serializer_round_trips_arbitrary_json_values(value: Any) -> None:
    serializer = PickleSerializer()
    serialized = serializer.serialize(value)
    assert serializer.deserialize(serialized) == value


@pytest.mark.parametrize(
    "value", [*PICKLE_LOOKALIKE_JSON_VALUES, *ARBITRARY_JSON_VALUES]
)
def test_json_serializer_serialized_value_deserialized_by_pickle_serializer(
    value: Any,
) -> None:
    json_ser = JsonSerializer()
    pickle_ser = PickleSerializer()
    serialized = json_ser.serialize(value)
    deserialized = pickle_ser.deserialize(serialized)
    assert deserialized == value


@pytest.mark.parametrize("value", PICKLE_ONLY_VALUES)
def test_pickle_serializer_round_trips_pickle_only_values(value: Any) -> None:
    serializer = PickleSerializer()
    serialized = serializer.serialize(value)
    # The serialized string should NOT be valid JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(serialized)
    deserialized = serializer.deserialize(serialized)
    assert deserialized == value


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_serializer_deserializes_all_pickle_protocols(protocol: int) -> None:
    value = CustomObject(name="protocol_test", count=protocol)
    payload = base64.b64encode(pickle.dumps(value, protocol=protocol)).decode("utf-8")
    serializer = PickleSerializer()
    assert serializer.deserialize(payload) == value


def test_pickle_serializer_never_unpickles_json_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_pickle_loads(data: bytes) -> Any:
        pytest.fail("pickle.loads must not be called for valid JSON payloads")

    monkeypatch.setattr(pickle, "loads", fail_on_pickle_loads)

    serializer = PickleSerializer()
    for value in [*PICKLE_LOOKALIKE_JSON_VALUES, *ARBITRARY_JSON_VALUES]:
        payload = serializer.serialize(value)
        assert serializer.deserialize(payload) == value


def test_pickle_serializer_rejects_invalid_payload() -> None:
    serializer = PickleSerializer()
    with pytest.raises((Exception,)):
        serializer.deserialize("not valid json or base64 pickle")


@pytest.mark.asyncio
async def test_workflow_context_restore_preserves_sw_eu_string() -> None:
    workflow = RememberWorkflow(serializer=PickleSerializer())
    ctx = Context(workflow)
    await workflow.run(ctx=ctx, text="SwEu")
    restored = Context.from_dict(workflow, ctx.to_dict())
    assert await restored.store.get("cell") == "SwEu"


@pytest.mark.asyncio
async def test_workflow_shared_context_second_run_preserves_sw_eu_string() -> None:
    workflow = RememberWorkflow(serializer=PickleSerializer())
    ctx = Context(workflow)
    await workflow.run(ctx=ctx, text="SwEu")
    assert await workflow.run(ctx=ctx, text="next") == "SwEu"
