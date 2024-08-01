import re


def convert_to_kebab_case(input_string: str) -> str:
    matches = re.findall(
        r"/[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+/g",
        input_string.lower(),
    )

    return "-".join(matches)
