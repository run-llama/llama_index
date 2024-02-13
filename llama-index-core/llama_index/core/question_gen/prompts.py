import json
from typing import Sequence

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.tools.types import ToolMetadata

# deprecated, kept for backward compatibility
SubQuestionPrompt = PromptTemplate


def build_tools_text(tools: Sequence[ToolMetadata]) -> str:
    tools_dict = {}
    for tool in tools:
        tools_dict[tool.name] = tool.description
    return json.dumps(tools_dict, indent=4)


PREFIX = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question:

"""


example_query_str = (
    "Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021"
)
example_tools = [
    ToolMetadata(
        name="uber_10k",
        description="Provides information about Uber financials for year 2021",
    ),
    ToolMetadata(
        name="lyft_10k",
        description="Provides information about Lyft financials for year 2021",
    ),
]
example_tools_str = build_tools_text(example_tools)
example_output = [
    SubQuestion(
        sub_question="What is the revenue growth of Uber", tool_name="uber_10k"
    ),
    SubQuestion(sub_question="What is the EBITDA of Uber", tool_name="uber_10k"),
    SubQuestion(
        sub_question="What is the revenue growth of Lyft", tool_name="lyft_10k"
    ),
    SubQuestion(sub_question="What is the EBITDA of Lyft", tool_name="lyft_10k"),
]
example_output_str = json.dumps({"items": [x.dict() for x in example_output]}, indent=4)

EXAMPLES = f"""\
# Example 1
<Tools>
```json
{example_tools_str}
```

<User Question>
{example_query_str}


<Output>
```json
{example_output_str}
```

""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

SUFFIX = """\
# Example 2
<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
"""

DEFAULT_SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX
