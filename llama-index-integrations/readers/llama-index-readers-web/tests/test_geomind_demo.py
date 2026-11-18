import pytest
from geomind_demo import extract_article_text


def test_extract_article_text():
    html = """
    <html>
        <nav>Home | Contact</nav>
        <article><p>White tea grows in Fuding.</p></article>
        <article><p>白茶产于福鼎。</p></article>
        <footer>Scan to order tea!</footer>
    </html>
    """

    text = extract_article_text(html)

    assert "White tea grows in Fuding." in text
    assert "白茶产于福鼎。" in text
    assert "Home | Contact" not in text
    assert "Scan to order tea!" not in text


def test_extract_article_text_without_articles():
    html = """
    <html>
        <nav>Home | Contact</nav>
        <p>This page has no article element.</p>
    </html>
    """

    with pytest.raises(ValueError, match="No article content found"):
        extract_article_text(html)
