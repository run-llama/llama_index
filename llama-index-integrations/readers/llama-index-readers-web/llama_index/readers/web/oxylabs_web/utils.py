import re
from typing import Any, Optional

from lxml.html import defs, fromstring, tostring
from lxml.html.clean import Cleaner


def clean_html(html: str) -> str:
    """Clean an HTML string."""
    cleaner = Cleaner(
        scripts=True,
        javascript=True,
        style=True,
        remove_tags=[],
        kill_tags=["nav", "svg", "footer", "noscript", "script", "form"],
        safe_attrs=[*list(defs.safe_attrs), "idx"],
        comments=True,
        inline_style=True,
        links=True,
        meta=False,
        page_structure=False,
        embedded=True,
        frames=False,
        forms=False,
        annoying_tags=False,
    )
    return cleaner.clean_html(html)  # type: ignore[no-any-return]


def strip_html(html: str) -> str:
    """
    Simplify an HTML string.

    Will remove unwanted elements, attributes, and redundant content
    Args:
        html (str): The input HTML string.

    Returns:
        str: The cleaned and simplified HTML string.

    """
    cleaned_html = clean_html(html)
    html_tree = fromstring(cleaned_html)

    for element in html_tree.iter():
        if "style" in element.attrib:
            del element.attrib["style"]

        if (
            (
                not element.attrib
                or (len(element.attrib) == 1 and "idx" in element.attrib)
            )
            and not element.getchildren()  # type: ignore[attr-defined]
            and (not element.text or not element.text.strip())
            and (not element.tail or not element.tail.strip())
        ):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    xpath_query = (
        ".//*[contains(@class, 'footer') or contains(@id, 'footer') or "
        "contains(@class, 'hidden') or contains(@id, 'hidden')]"
    )
    elements_to_remove = html_tree.xpath(xpath_query)
    for element in elements_to_remove:  # type: ignore[assignment, union-attr]
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    stripped_html = tostring(html_tree, encoding="unicode")
    stripped_html = re.sub(r"\s{2,}", " ", stripped_html)
    return re.sub(r"\n{2,}", "", stripped_html)


def json_to_markdown(data: Any, level: int = 0, header: Optional[str] = None) -> str:
    """
    Recursively converts a Python object (from JSON) into a Markdown string.

    Args:
        data: The Python object to convert.
        level: The current nesting level (used for indentation and heading levels).
        header: Section header.

    Returns:
        A string containing the Markdown representation of the data.

    """
    markdown_parts = []
    indent = "  " * level

    if isinstance(data, dict):
        for key, value in data.items():
            heading_level = min(level + 1, 6)
            markdown_parts.append(f"{indent}{'#' * heading_level} {key}\n")
            markdown_parts.append(json_to_markdown(value, level + 1))
            markdown_parts.append("\n")

    elif isinstance(data, list):
        if not data:
            markdown_parts.append(f"{indent}- *Empty List*\n")
        else:
            if header:
                markdown_parts.append(f"{indent}- *{header}*\n")
            for index, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    markdown_parts.append(f"{indent}- Item {index + 1}:\n")
                    markdown_parts.append(json_to_markdown(item, level + 1))
                else:
                    markdown_parts.append(f"{indent}- {item!s}\n")

    elif isinstance(data, str):
        if "\n" in data:
            cleaned_data = data.replace("\n", "\n" + indent + "> ")
            markdown_parts.append(f"{indent}> {cleaned_data}\n")
        else:
            markdown_parts.append(f"{indent}{data}\n")

    elif isinstance(data, (int, float, bool)) or data is None:
        markdown_parts.append(f"{indent}{data!s}\n")

    else:
        markdown_parts.append(f"{indent}{data!s}\n")

    return "".join(markdown_parts).rstrip("\n") + "\n"
