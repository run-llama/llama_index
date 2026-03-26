import pytest
from llama_index.readers.confluence.html_parser import HtmlTextParser


class TestHtmlTextParser:
    def test_parser_initialization(self):
        parser = HtmlTextParser()
        assert parser is not None

    @pytest.mark.parametrize(
        ("html_input", "expected_contains"),
        [
            ("<p>Simple text</p>", "Simple text"),
            ("<p>First paragraph</p><p>Second paragraph</p>", "First paragraph"),
            ("<h1>Main Title</h1>", "Main Title"),
            ("<h2>Subtitle</h2>", "Subtitle"),
            ("<h3>Section</h3>", "Section"),
            ("<strong>Bold text</strong>", "Bold text"),
            ("<b>Bold text</b>", "Bold text"),
            ("<em>Italic text</em>", "Italic text"),
            ("<i>Italic text</i>", "Italic text"),
            ("<code>inline code</code>", "inline code"),
        ],
    )
    def test_basic_formatting(self, html_input, expected_contains):
        parser = HtmlTextParser()
        result = parser.convert(html_input)
        assert expected_contains in result

    @pytest.mark.parametrize(
        ("html_input", "expected_marker"),
        [
            ("<h1>Title</h1>", "#"),
            ("<h2>Subtitle</h2>", "##"),
            ("<h3>Section</h3>", "###"),
            ("<strong>Bold</strong>", "**"),
            ("<b>Bold</b>", "**"),
            ("<em>Italic</em>", "*"),
        ],
    )
    def test_markdown_style_conversion(self, html_input, expected_marker):
        parser = HtmlTextParser()
        result = parser.convert(html_input)
        assert expected_marker in result

    @pytest.mark.parametrize(
        ("html_input", "expected_contains"),
        [
            ("<ul><li>Item 1</li><li>Item 2</li></ul>", "Item 1"),
            ("<ol><li>First</li><li>Second</li></ol>", "First"),
            (
                "<ul><li>Parent<ul><li>Child</li></ul></li></ul>",
                "Parent",
            ),
        ],
    )
    def test_lists(self, html_input, expected_contains):
        parser = HtmlTextParser()
        result = parser.convert(html_input)
        assert expected_contains in result

    def test_unordered_list_structure(self):
        parser = HtmlTextParser()
        html = "<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>"
        result = parser.convert(html)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Item 3" in result
        assert any(marker in result for marker in ["*", "-", "+"])

    def test_ordered_list_structure(self):
        parser = HtmlTextParser()
        html = "<ol><li>First item</li><li>Second item</li></ol>"
        result = parser.convert(html)
        assert "First item" in result
        assert "Second item" in result

    @pytest.mark.parametrize(
        ("html_input", "expected_text", "expected_marker"),
        [
            ('<a href="https://example.com">Link text</a>', "Link text", "example.com"),
            ('<a href="/page">Internal link</a>', "Internal link", "/page"),
            ('<a href="url"><strong>Bold link</strong></a>', "Bold link", "url"),
        ],
    )
    def test_links(self, html_input, expected_text, expected_marker):
        parser = HtmlTextParser()
        result = parser.convert(html_input)
        assert expected_text in result
        assert expected_marker in result

    def test_link_markdown_format(self):
        parser = HtmlTextParser()
        html = '<a href="https://example.com">Example</a>'
        result = parser.convert(html)
        assert "[Example](https://example.com)" in result

    @pytest.mark.parametrize(
        ("html_input", "expected_contains"),
        [
            ("<pre>Code block</pre>", "Code block"),
            ("<pre><code>def foo():</code></pre>", "def foo():"),
            ("<p>Use <code>print()</code> function</p>", "print()"),
        ],
    )
    def test_code_blocks(self, html_input, expected_contains):
        parser = HtmlTextParser()
        result = parser.convert(html_input)
        assert expected_contains in result

    def test_table_basic(self):
        parser = HtmlTextParser()
        html = """
        <table>
            <thead>
                <tr><th>Header 1</th><th>Header 2</th></tr>
            </thead>
            <tbody>
                <tr><td>Cell 1</td><td>Cell 2</td></tr>
            </tbody>
        </table>
        """
        result = parser.convert(html)
        assert "Header 1" in result
        assert "Header 2" in result
        assert "Cell 1" in result
        assert "Cell 2" in result

    def test_table_multiple_rows(self):
        parser = HtmlTextParser()
        html = """
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Alice</td><td>30</td></tr>
            <tr><td>Bob</td><td>25</td></tr>
        </table>
        """
        result = parser.convert(html)
        assert "Name" in result
        assert "Age" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "30" in result
        assert "25" in result

    def test_mixed_formatting(self):
        parser = HtmlTextParser()
        html = """
        <h1>Main Title</h1>
        <p>This is a paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
        <ul>
            <li>First item</li>
            <li>Second item with <code>code</code></li>
        </ul>
        <p>A <a href="https://example.com">link</a> in a paragraph.</p>
        """
        result = parser.convert(html)
        assert "Main Title" in result
        assert "paragraph" in result
        assert "bold" in result
        assert "italic" in result
        assert "First item" in result
        assert "Second item" in result
        assert "code" in result
        assert "[link](https://example.com)" in result

    def test_nested_lists(self):
        parser = HtmlTextParser()
        html = """
        <ul>
            <li>Parent item 1
                <ul>
                    <li>Child item 1</li>
                    <li>Child item 2</li>
                </ul>
            </li>
            <li>Parent item 2</li>
        </ul>
        """
        result = parser.convert(html)
        assert "Parent item 1" in result
        assert "Child item 1" in result
        assert "Child item 2" in result
        assert "Parent item 2" in result

    def test_empty_paragraph(self):
        parser = HtmlTextParser()
        html = "<p></p>"
        result = parser.convert(html)
        assert result is not None

    def test_empty_string(self):
        parser = HtmlTextParser()
        result = parser.convert("")
        assert result is not None

    def test_special_characters(self):
        parser = HtmlTextParser()
        html = "<p>&lt;div&gt; &amp; &quot;test&quot;</p>"
        result = parser.convert(html)
        assert result is not None
        assert len(result) > 0

    def test_line_breaks(self):
        parser = HtmlTextParser()
        html = "<p>Line 1<br/>Line 2<br/>Line 3</p>"
        result = parser.convert(html)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_div_elements(self):
        parser = HtmlTextParser()
        html = "<div>Content in div</div>"
        result = parser.convert(html)
        assert "Content in div" in result

    def test_nested_formatting(self):
        parser = HtmlTextParser()
        html = "<p><strong><em>Bold and italic</em></strong></p>"
        result = parser.convert(html)
        assert "Bold and italic" in result

    def test_confluence_style_content(self):
        parser = HtmlTextParser()
        html = """
        <h1>Page Title</h1>
        <p>Introduction paragraph with <strong>important</strong> information.</p>
        <h2>Section 1</h2>
        <p>Section content with a <a href="/link">link</a>.</p>
        <ul>
            <li>Bullet point 1</li>
            <li>Bullet point 2</li>
        </ul>
        <h2>Section 2</h2>
        <p>Code example: <code>function()</code></p>
        <pre><code>def example():
    return True</code></pre>
        """
        result = parser.convert(html)
        assert "Page Title" in result
        assert "Introduction" in result
        assert "important" in result
        assert "Section 1" in result
        assert "Section 2" in result
        assert "link" in result
        assert "Bullet point 1" in result
        assert "function()" in result
        assert "def example():" in result

    def test_whitespace_handling(self):
        parser = HtmlTextParser()
        html = "<p>Text   with   multiple   spaces</p>"
        result = parser.convert(html)
        assert "Text" in result
        assert "spaces" in result

    def test_convert_returns_string(self):
        parser = HtmlTextParser()
        result = parser.convert("<p>Test</p>")
        assert isinstance(result, str)

    def test_multiple_headings(self):
        parser = HtmlTextParser()
        html = """
        <h1>H1 Title</h1>
        <h2>H2 Subtitle</h2>
        <h3>H3 Section</h3>
        <h4>H4 Subsection</h4>
        """
        result = parser.convert(html)
        assert "H1 Title" in result
        assert "H2 Subtitle" in result
        assert "H3 Section" in result
        assert "H4 Subsection" in result
