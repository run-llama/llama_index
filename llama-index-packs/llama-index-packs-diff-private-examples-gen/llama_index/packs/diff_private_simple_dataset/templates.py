from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

ZERO_SHOT_TEMPLATE = "{instruction}\n{label_heading}: {label}"

USER_TEMPLATE = "{instruction}\n{label_heading}: {label}"

ASSISTANT_TEMPLATE = "{label_heading}: {label}\n{text_heading}: {synthetic_text}"

FIRST_SHOT_TEMPLATE = "{label_heading}: {example_label}\n{text_heading}: {example_text}"

SECOND_SHOT_TEMPLATE = (
    "{label_heading}: {second_example_label}\n{text_heading}: {second_example_text}"
)

THIRD_SHOT_TEMPLATE = (
    "{label_heading}: {third_example_label}\n{text_heading}: {third_example_text}"
)

message_templates = [
    ChatMessage(
        content="You are a helpful assistant that follows instructions and formatting strictly.",
        role=MessageRole.SYSTEM,
    ),
]

refine_chat_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            content="You are an editor. Refine the assitant's response to better conform to the user provided instructions.",
            role=MessageRole.SYSTEM,
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

three_shot_chat_template = ChatPromptTemplate(
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
            content=THIRD_SHOT_TEMPLATE,
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

ZERO_SHOT_COMPLETION_TEMPLATE = (
    "{instruction}\n" "{label_heading}: {label}\n{text_heading}: {synthetic_text}"
)
zero_shot_completion_template = PromptTemplate(ZERO_SHOT_COMPLETION_TEMPLATE)

SINGLE_EXAMPLE_TEMPLATE = (
    "{label_heading}: {example_label}\n{text_heading}: {example_text}\n\n"
)
single_example_template = PromptTemplate(SINGLE_EXAMPLE_TEMPLATE)


FEW_SHOT_COMPLETION_TEMPLATE = (
    "{instruction}\n\n"
    "{few_shot_examples}"
    "{label_heading}: {label}\n{text_heading}: {synthetic_text}"
)
few_shot_completion_template = PromptTemplate(FEW_SHOT_COMPLETION_TEMPLATE)

THREE_SHOT_COMPLETION_TEMPLATE = (
    "{instruction}\n"
    "{label_heading}: {example_label}\n{text_heading}: {example_text}\n\n"
    "{label_heading}: {second_example_label}\n{text_heading}: {second_example_text}\n\n"
    "{label_heading}: {third_example_label}\n{text_heading}: {third_example_text}\n\n"
    "{label_heading}: {label}\n{text_heading}: {synthetic_text}"
)
three_shot_completion_template = PromptTemplate(THREE_SHOT_COMPLETION_TEMPLATE)
