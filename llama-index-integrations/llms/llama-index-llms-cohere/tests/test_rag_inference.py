import pytest

from llama_index.core.prompts import MessageRole
from llama_index.llms.cohere import DocumentMessage
from llama_index.llms.cohere.utils import document_message_to_cohere_document

text1 = "birds flying high"
text2 = "sun in the sky"
text3 = "breeze driftin' on by"
text4 = "fish in the sea"
text5 = "river running free"
texts = [text1, text2, text3, text4, text5]


@pytest.mark.parametrize(
    "message, expected",  # noqa: PT006
    [
        pytest.param(
            DocumentMessage(
                role=MessageRole.USER,
                content="\n\n".join(
                    [f"file_path: nina.txt\n\n{text}" for text in texts]
                ),
                additional_kwargs={},
            ),
            [{"file_path": "nina.txt", "text": text} for text in texts],
            id="single field, multiple documents",
        ),
        pytest.param(
            DocumentMessage(
                role=MessageRole.USER,
                content="\n\n".join(
                    [
                        f"file_path: nina.txt\n\nfile_name: greatest-hits\n\n{text}"
                        for text in texts
                    ]
                ),
                additional_kwargs={},
            ),
            [
                {"file_path": "nina.txt", "file_name": "greatest-hits", "text": text}
                for text in texts
            ],
            id="multiple fields (same count), multiple documents",
        ),
        pytest.param(
            DocumentMessage(
                role=MessageRole.USER,
                content="\n\n".join(texts),
                additional_kwargs={},
            ),
            [{"text": "\n\n".join(texts)}],
            id="no fields (just text), multiple documents",
        ),
    ],
)
def test_document_message_to_cohere_document(message, expected):
    res = document_message_to_cohere_document(message)
    assert res == expected
