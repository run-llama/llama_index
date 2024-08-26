from unittest import mock

import pytest

from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.workflow.events import Event


def test_send_event_step_is_none(session):
    session._queues = {"step1": mock.MagicMock(), "step2": mock.MagicMock()}
    ev = Event(foo="bar")
    session.send_event(ev)
    for q in session._queues.values():
        q.put_nowait.assert_called_with(ev)
    assert session._broker_log == [ev]


def test_send_event_to_non_existent_step(session):
    with pytest.raises(
        WorkflowRuntimeError, match="Step does_not_exist does not exist"
    ):
        session.send_event(Event(), "does_not_exist")


def test_send_event_to_wrong_step(session):
    session._workflow._get_steps = mock.MagicMock(
        return_value={"step": mock.MagicMock()}
    )

    with pytest.raises(
        WorkflowRuntimeError,
        match="Step step does not accept event of type <class 'llama_index.core.workflow.events.Event'>",
    ):
        session.send_event(Event(), "step")


def test_send_event_to_step(session):
    step2 = mock.MagicMock()
    step2.__step_config.accepted_events = [Event]

    session._workflow._get_steps = mock.MagicMock(
        return_value={"step1": mock.MagicMock(), "step2": step2}
    )
    session._queues = {"step1": mock.MagicMock(), "step2": mock.MagicMock()}

    ev = Event(foo="bar")
    session.send_event(ev, "step2")

    session._queues["step1"].put_nowait.assert_not_called()
    session._queues["step2"].put_nowait.assert_called_with(ev)


def test_get_result(session):
    session._retval = 42
    assert session.get_result() == 42


def test_get_context(session):
    ctx = session.get_context("step")
    assert session._step_to_context["step"] == ctx
    assert ctx._parent == session._root_context
