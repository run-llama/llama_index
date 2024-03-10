from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

ZERO_SHOT_TEMPLATE = "{label_heading}: {label}\n" "{text_heading}: {synthetic_text}"

ONE_SHOT_TEMPLATE = (
    "{label_heading}: {example_label}\n"
    "{text_heading}: {example_text}"
    "\n\n"
    "{label_heading}: {label}\n"
    "{text_heading}: {synthetic_text}"
)

TWO_SHOT_TEMPLATE = (
    "{label_heading}: {example1_label}\n"
    "{text_heading}: {example1_text}"
    "\n\n"
    "{label_heading}: {example2_label}\n"
    "{text_heading}: {example2_text}"
    "\n\n"
    "{label_heading}: {label}\n"
    "{text_heading}: {synthetic_text}"
)

message_templates = [
    ChatMessage(content="{instruction}", role=MessageRole.SYSTEM),
]

zero_shot_chat_template = ChatPromptTemplate(
    message_templates=[
        *message_templates,
        ChatMessage(
            content=ZERO_SHOT_TEMPLATE,
            role=MessageRole.USER,
        ),
    ]
)

one_shot_chat_template = ChatPromptTemplate(
    message_templates=[
        *message_templates,
        ChatMessage(
            content=ONE_SHOT_TEMPLATE,
            role=MessageRole.USER,
        ),
    ]
)

two_shot_chat_template = ChatPromptTemplate(
    message_templates=[
        *message_templates,
        ChatMessage(
            content=TWO_SHOT_TEMPLATE,
            role=MessageRole.USER,
        ),
    ]
)
