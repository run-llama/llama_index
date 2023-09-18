from llama_index.text_splitter.html_splitter import HTMLSplitter


def test_no_splits() -> None:
    splitter = HTMLSplitter(tags=["h2"])

    splits = splitter.split_text(
        """
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
    print(splits)
    assert len(splits) == 0


def test_single_splits() -> None:
    splitter = HTMLSplitter(tags=["h1"])

    splits = splitter.split_text(
        """
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
    assert len(splits) == 1
    assert splits[0] == "This is the Title"


def test_multiple_tags_splits() -> None:
    splitter = HTMLSplitter(tags=["h2", "p"])

    splits = splitter.split_text(
        """
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
    assert len(splits) == 3
    assert splits[0] == "This is a paragraph of text."
    assert splits[1] == "Section 1"
    assert splits[2] == "This is the first paragraph."
