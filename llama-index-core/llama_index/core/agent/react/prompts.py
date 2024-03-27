"""Default prompt for ReAct agent."""

# ReAct chat prompt
# TODO: have formatting instructions be a part of react output parser
from pathlib import Path

p = Path(__file__).with_name("system_headaer_template.md")
with p.open("r") as f:
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
