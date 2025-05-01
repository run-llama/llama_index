import re
from typing import Any, List, Optional, Pattern

from redisvl.query.filter import Tag, Num, Text


class TokenEscaper:
    """
    Escape punctuation within an input string. Taken from RedisOM Python.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        def escape_symbol(match: re.Match) -> str:
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)


# Global constant defining field specifications
REDIS_LLAMA_FIELD_SPEC = {
    "tag": {
        "class": Tag,
        "operators": {
            "==": lambda f, v: f == v,
            "!=": lambda f, v: f != v,
            "in": lambda f, v: f == v,
            "nin": lambda f, v: f != v,
            "contains": lambda f, v: f == v,
        },
    },
    "numeric": {
        "class": Num,
        "operators": {
            "==": lambda f, v: f == v,
            "!=": lambda f, v: f != v,
            ">": lambda f, v: f > v,
            "<": lambda f, v: f < v,
            ">=": lambda f, v: f >= v,
            "<=": lambda f, v: f <= v,
        },
    },
    "text": {
        "class": Text,
        "operators": {
            "==": lambda f, v: f == v,
            "!=": lambda f, v: f != v,
            "text_match": lambda f, v: f % v,
        },
    },
}
