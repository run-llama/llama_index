import importlib.util
import os
import tempfile
from pathlib import Path

import pytest
from llama_index.readers.file import HTMLTagReader


@pytest.fixture()
def html_str() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HTML Sections Example</title>
</head>
<body>
    <header>
        <h1>Welcome to My Website</h1>
    </header>

    <nav>
        <ul>
            <li><a href="#">Home</a></li>
            <li><a href="#">About</a></li>
            <li><a href="#">Services</a></li>
            <li><a href="#">Contact</a></li>
        </ul>
    </nav>

    <section id="about">
        <h2>About Us</h2>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
    </section>

    <section id="services">
        <h2>Our Services</h2>
        <ul>
            <li>Service 1</li>
            <li>Service 2</li>
            <li>Service 3</li>
        </ul>
    </section>

    <section>
        <h2>Contact Us</h2>
        <p>You can reach us at \
<a href="mailto:contact@example.com">contact@example.com</a>.</p>
    </section>

    <footer>
        <p>&copy; 2023 My Website</p>
    </footer>
</body>
</html>
"""


@pytest.mark.xfail(
    raises=ImportError,
    reason="Requires beautifulsoup4.",
    condition=importlib.util.find_spec("beautifulsoup4") is None,
)
def test_html_tag_reader(html_str: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".html"
    ) as temp_file:
        temp_file.write(html_str)
        temp_file_path = Path(temp_file.name)

    reader = HTMLTagReader(ignore_no_id=True)
    docs = reader.load_data(temp_file_path)
    assert len(docs) == 2
    assert docs[0].metadata["tag_id"] == "about"
    assert docs[1].metadata["tag_id"] == "services"

    reader = HTMLTagReader()
    docs = reader.load_data(temp_file_path)
    assert len(docs) == 3
    assert docs[2].metadata["tag_id"] is None

    os.remove(temp_file.name)


@pytest.mark.parametrize("tag", ["section", "article"])
@pytest.mark.parametrize("wrapper", ["<div>{}</div>", "<div><div>{}</div></div>"])
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Before <b>bold</b>{nested} after", "Parent text\nBefore bold after"),
        ("{nested}", "Parent text"),
    ],
)
def test_html_tag_reader_excludes_wrapped_nested_tags(
    tmp_path: Path, tag: str, wrapper: str, content: str, expected: str
) -> None:
    nested = f'<{tag} id="child">Child text</{tag}>'
    content = wrapper.format(content.format(nested=nested))
    file = tmp_path / "nested.html"
    file.write_text(
        f'<{tag} id="parent"><p>Parent text</p>{content}</{tag}>', encoding="utf-8"
    )

    docs = HTMLTagReader(tag=tag).load_data(file, extra_info={"source": "test"})

    assert [doc.text for doc in docs] == [expected, "Child text"]
    assert [doc.metadata["tag_id"] for doc in docs] == ["parent", "child"]
    assert all(doc.metadata["source"] == "test" for doc in docs)
