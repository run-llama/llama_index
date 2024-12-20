from llama_index.core.llama_dataset.simple import (
    LabelledSimpleDataExample,
)
from llama_index.packs.diff_private_simple_dataset.templates import (
    few_shot_completion_template,
    single_example_template,
)
from llama_index.packs.diff_private_simple_dataset.base import PromptBundle
from functools import reduce


def test_few_shot_template():
    # arrange
    prompt_bundle = PromptBundle(
        instruction="INSTRUCTION", label_heading="LABEL", text_heading="TEXT"
    )
    examples = [
        LabelledSimpleDataExample(reference_label="X", text="test x"),
        LabelledSimpleDataExample(reference_label="Y", text="test y"),
        LabelledSimpleDataExample(reference_label="Z", text="test z"),
    ]
    single_templates = [
        single_example_template.format(
            label_heading=prompt_bundle.label_heading,
            text_heading=prompt_bundle.text_heading,
            example_label=x.reference_label,
            example_text=x.text,
        )
        for x in examples
    ]
    synthetic_text = ""
    label = "X"

    # act
    few_shot_examples = reduce(lambda x, y: x + y, single_templates)
    prompt = few_shot_completion_template.format(
        instruction=prompt_bundle.instruction,
        label_heading=prompt_bundle.label_heading,
        text_heading=prompt_bundle.text_heading,
        few_shot_examples=few_shot_examples,
        label=label,
        synthetic_text=synthetic_text,
    )

    # assert
    expected = (
        "INSTRUCTION\n\n"
        "LABEL: X\n"
        "TEXT: test x\n\n"
        "LABEL: Y\n"
        "TEXT: test y\n\n"
        "LABEL: Z\n"
        "TEXT: test z\n\n"
        "LABEL: X\n"
        "TEXT: "
    )
    assert expected == prompt
