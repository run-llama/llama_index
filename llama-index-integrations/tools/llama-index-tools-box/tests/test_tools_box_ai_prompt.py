import os
import pytest
from box_sdk_gen import BoxClient
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.box import BoxAIPromptToolSpec
from tests.conftest import get_testing_data


def test_box_tool_ai_prompt(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    box_tool = BoxAIPromptToolSpec(box_client=box_client_ccg_integration_testing)

    doc = box_tool.ai_prompt(
        file_id=test_data["test_ppt_id"], ai_prompt="Summarize the document"
    )

    assert doc.text is not None


@pytest.mark.asyncio
async def test_box_tool_ai_prompt_agent(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    document_id = test_data["test_ppt_id"]
    openai_api_key = test_data["openai_api_key"]
    ai_prompt = "Summarize the document"

    if openai_api_key is None:
        raise pytest.skip("OpenAI API key is not provided.")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    box_tool = BoxAIPromptToolSpec(box_client=box_client_ccg_integration_testing)

    agent = FunctionAgent(
        tools=box_tool.to_tool_list(),
        llm=OpenAI(model="gpt-4.1"),
    )

    answer = await agent.run(f"{ai_prompt} for {document_id}")
    # print(answer)
    assert answer is not None
