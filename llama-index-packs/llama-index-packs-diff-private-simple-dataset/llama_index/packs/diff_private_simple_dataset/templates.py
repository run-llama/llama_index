from llama_index.core import PromptTemplate

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
