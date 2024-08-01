"""Default prompt for ReAct agent."""
from pathlib import Path

# TODO: have formatting instructions be a part of react output parser
with (
    Path(__file__).parents[0] / Path("templates") / Path("system_header_template.md")
).open("r") as f:
    __BASE_REACT_CHAT_SYSTEM_HEADER = f.read()

REACT_CHAT_SYSTEM_HEADER = __BASE_REACT_CHAT_SYSTEM_HEADER.replace(
    "{context_prompt}", "", 1
)

CONTEXT_REACT_CHAT_SYSTEM_HEADER = __BASE_REACT_CHAT_SYSTEM_HEADER.replace(
    "{context_prompt}",
    """
Here is some context to help you answer the question and plan:
{context}
""",
    1,
)
