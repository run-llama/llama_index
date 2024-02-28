import importlib.util

import pytest
from llama_index.legacy.node_parser.file.html import HTMLNodeParser
from llama_index.legacy.schema import Document


@pytest.mark.xfail(
    raises=ImportError,
    reason="Requires beautifulsoup4.",
    condition=importlib.util.find_spec("beautifulsoup4") is None,
)
def test_no_splits() -> None:
    html_parser = HTMLNodeParser(tags=["h2"])

    splits = html_parser.get_nodes_from_documents(
        [
            Document(
                text="""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1 id="title">This is the Title</h1>
    <p>This is a paragraph of text.</p>
</body>
</html>
    """
            )
        ]
    )
    print(splits)
    assert len(splits) == 0


@pytest.mark.xfail(
    raises=ImportError,
    reason="Requires beautifulsoup4.",
    condition=importlib.util.find_spec("beautifulsoup4") is None,
)
def test_single_splits() -> None:
    html_parser = HTMLNodeParser(tags=["h1"])

    splits = html_parser.get_nodes_from_documents(
        [
            Document(
                text="""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1 id="title">This is the Title</h1>
    <p>This is a paragraph of text.</p>
</body>
</html>
    """
            )
        ]
    )
    assert len(splits) == 1
    assert splits[0].text == "This is the Title"
    assert splits[0].metadata["tag"] == "h1"


@pytest.mark.xfail(
    raises=ImportError,
    reason="Requires beautifulsoup4.",
    condition=importlib.util.find_spec("beautifulsoup4") is None,
)
def test_multiple_tags_splits() -> None:
    html_parser = HTMLNodeParser(tags=["h2", "p"])

    splits = html_parser.get_nodes_from_documents(
        [
            Document(
                text="""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1 id="title">This is the Title</h1>
    <p>This is a paragraph of text.</p>
    <div>
        <h2 id="section1">Section 1</h2>
    </div>
    <p>This is the first paragraph.</p>
</body>
</html>
    """
            )
        ]
    )
    assert len(splits) == 3
    assert splits[0].text == "This is a paragraph of text."
    assert splits[1].text == "Section 1"
    assert splits[2].text == "This is the first paragraph."
    assert splits[0].metadata["tag"] == "p"
    assert splits[1].metadata["tag"] == "h2"
    assert splits[2].metadata["tag"] == "p"


@pytest.mark.xfail(
    raises=ImportError,
    reason="Requires beautifulsoup4.",
    condition=importlib.util.find_spec("beautifulsoup4") is None,
)
def test_nesting_tags_splits() -> None:
    html_parser = HTMLNodeParser(tags=["h2", "b"])

    splits = html_parser.get_nodes_from_documents(
        [
            Document(
                text="""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1 id="title">This is the Title</h1>
    <p>This is a paragraph of text.</p>
    <div>
        <h2 id="section1">Section 1 <b>bold</b></h2>
    </div>
    <p>This is the first paragraph.</p>
</body>
</html>
    """
            )
        ]
    )
    assert len(splits) == 2
    assert splits[0].text == "Section 1"
    assert splits[1].text == "bold"
    assert splits[0].metadata["tag"] == "h2"
    assert splits[1].metadata["tag"] == "b"


@pytest.mark.xfail(
    raises=ImportError,
    reason="Requires beautifulsoup4.",
    condition=importlib.util.find_spec("beautifulsoup4") is None,
)
def test_neighbor_tags_splits() -> None:
    html_parser = HTMLNodeParser(tags=["p"])

    splits = html_parser.get_nodes_from_documents(
        [
            Document(
                text="""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <p>This is the first paragraph.</p>
    <p>This is the second paragraph</p>
</body>
</html>
    """
            )
        ]
    )
    assert len(splits) == 1
