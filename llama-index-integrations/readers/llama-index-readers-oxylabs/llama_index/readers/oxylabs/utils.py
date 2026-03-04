from typing import Any, Optional


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
                markdown_parts.append(f"# {header}\n")

            for index, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    markdown_parts.append(f"{indent}- Item {index + 1}:\n")
                    markdown_parts.append(json_to_markdown(item, level + 1))
                else:
                    markdown_parts.append(f"{indent}- {item!s}\n")

    elif isinstance(data, str):
        if "\n" in data:
            # nl var to enable the usage of this symbol inside f-string expressions
            nl = "\n"

            markdown_parts.append(f"{indent}> {data.replace(nl, nl + indent + '> ')}\n")
        else:
            markdown_parts.append(f"{indent}{data}\n")

    elif isinstance(data, (int, float, bool)) or data is None:
        markdown_parts.append(f"{indent}{data!s}\n")

    else:
        markdown_parts.append(f"{indent}{data!s}\n")

    return "".join(markdown_parts).rstrip("\n") + "\n"
