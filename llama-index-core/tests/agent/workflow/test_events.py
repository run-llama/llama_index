from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.memory import Memory


def test_agent_workflow_start_event():
    event = AgentWorkflowStartEvent(
        user_msg="Hello, world!",
        chat_history=[ChatMessage(role="user", content="Hello, world!")],
        max_iterations=10,
    )
    assert event.user_msg == "Hello, world!"
    assert event.chat_history[0].role.value == "user"
    assert event.chat_history[0].content == "Hello, world!"
    assert event.max_iterations == 10


def test_agent_workflow_start_event_with_dict():
    event = AgentWorkflowStartEvent(
        user_msg="Hello, world!",
        chat_history=[{"role": "user", "content": "Hello, world!"}],
        max_iterations=10,
    )
    assert event.user_msg == "Hello, world!"
    assert event.chat_history[0].role.value == "user"
    assert event.chat_history[0].content == "Hello, world!"
    assert event.max_iterations == 10


def test_agent_workflow_start_event_to_dict():
    event = AgentWorkflowStartEvent(
        user_msg="Hello, world!",
        chat_history=[ChatMessage(role="user", content="Hello, world!")],
        max_iterations=10,
        memory=Memory.from_defaults(),
    )

    # Memory is not included in the dump
    dump = event.model_dump()
    assert len(dump) == 3
    assert dump["user_msg"] == "Hello, world!"
    assert dump["chat_history"][0]["role"] == "user"
    assert dump["chat_history"][0]["blocks"][0]["text"] == "Hello, world!"
    assert dump["max_iterations"] == 10
