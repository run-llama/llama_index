from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.bgpt import BGPTToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in BGPTToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@patch("llama_index.tools.bgpt.base.requests.post")
def test_search_papers(mock_post: MagicMock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "results": [
                {
                    "title": "Example study",
                    "doi": "10.1000/example",
                    "methods_and_experimental_techniques": "RCT",
                    "paper_limitations_and_biases": "Small sample",
                }
            ]
        },
    )
    tool = BGPTToolSpec()
    docs = tool.search_papers("CRISPR", num_results=1)
    assert len(docs) == 1
    assert "Example study" in docs[0].text
    assert docs[0].metadata["doi"] == "10.1000/example"
    mock_post.assert_called_once()


@patch("llama_index.tools.bgpt.base.requests.post")
def test_lookup_paper(mock_post: MagicMock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "result": {
                "title": "DOI paper",
                "doi": "10.1000/doi",
                "how_to_falsify": "Repeat with larger cohort",
            }
        },
    )
    tool = BGPTToolSpec()
    docs = tool.lookup_paper("10.1000/doi")
    assert len(docs) == 1
    assert "how_to_falsify" in docs[0].text
