from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

ZERO_SHOT_TEMPLATE = "{instruction}\nn{label_heading}: {label}"

USER_TEMPLATE = "{instruction}\n{label_heading}: {label}"

ASSISTANT_TEMPLATE = "{label_heading}: {label}\n{text_heading}: {synthetic_text}"

FIRST_SHOT_TEMPLATE = (
    "{label_heading}: {example_label}\n" "{text_heading}: {example_text}"
)

SECOND_SHOT_TEMPLATE = (
    "{label_heading}: {second_example_label}\n" "{text_heading}: {second_example_text}"
)

message_templates = [
    ChatMessage(content="You are a helpful assistant.", role=MessageRole.SYSTEM),
]

zero_shot_chat_template = ChatPromptTemplate(
    message_templates=[
        *message_templates,
        ChatMessage(
            content=USER_TEMPLATE,
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=ASSISTANT_TEMPLATE,
            role=MessageRole.ASSISTANT,
        ),
    ]
)

one_shot_chat_template = ChatPromptTemplate(
    message_templates=[
        *message_templates,
        ChatMessage(
            content=USER_TEMPLATE,
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=FIRST_SHOT_TEMPLATE,
            role=MessageRole.ASSISTANT,
        ),
        ChatMessage(
            content=USER_TEMPLATE,
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=ASSISTANT_TEMPLATE,
            role=MessageRole.ASSISTANT,
        ),
    ]
)

two_shot_chat_template = ChatPromptTemplate(
    message_templates=[
        *message_templates,
        ChatMessage(
            content=USER_TEMPLATE,
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=FIRST_SHOT_TEMPLATE,
            role=MessageRole.ASSISTANT,
        ),
        ChatMessage(
            content=USER_TEMPLATE,
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=SECOND_SHOT_TEMPLATE,
            role=MessageRole.ASSISTANT,
        ),
        ChatMessage(
            content=USER_TEMPLATE,
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=ASSISTANT_TEMPLATE,
            role=MessageRole.ASSISTANT,
        ),
    ]
)
