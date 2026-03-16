from llama_index.core import PromptTemplate
from llama_index.core.agent.workflow import ReActAgent

from textwrap import dedent


def test_partial_formatted_system_prompt():
    """Partially formatted context should be preserved."""
    agent = ReActAgent()

    prompt = PromptTemplate(
        dedent(
            """\
            Required template variables:
            {tool_desc}
            {tool_names}

            Additional variables:
            {dummy_var}
            """
        )
    )

    dummy_var = "dummy_context"
    agent.update_prompts({"react_header": prompt.partial_format(dummy_var=dummy_var)})

    assert dummy_var in agent.formatter.system_header
