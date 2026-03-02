# Placeholder classes for filter types (will be implemented with actual Valkey filter logic)
class Tag:
    """Tag field filter class."""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: any) -> "TagFilter":
        return TagFilter(self.name, "==", value)

    def __ne__(self, value: any) -> "TagFilter":
        return TagFilter(self.name, "!=", value)


class Num:
    """Numeric field filter class."""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: any) -> "NumFilter":
        return NumFilter(self.name, "==", value)

    def __ne__(self, value: any) -> "NumFilter":
        return NumFilter(self.name, "!=", value)

    def __gt__(self, value: any) -> "NumFilter":
        return NumFilter(self.name, ">", value)

    def __lt__(self, value: any) -> "NumFilter":
        return NumFilter(self.name, "<", value)

    def __ge__(self, value: any) -> "NumFilter":
        return NumFilter(self.name, ">=", value)

    def __le__(self, value: any) -> "NumFilter":
        return NumFilter(self.name, "<=", value)


class Text:
    """Text field filter class."""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: any) -> "TextFilter":
        return TextFilter(self.name, "==", value)

    def __ne__(self, value: any) -> "TextFilter":
        return TextFilter(self.name, "!=", value)

    def __mod__(self, value: any) -> "TextFilter":
        return TextFilter(self.name, "text_match", value)


class TagFilter:
    """Tag filter expression."""

    def __init__(self, name: str, operator: str, value: any):
        self.name = name
        self.operator = operator
        self.value = value


class NumFilter:
    """Numeric filter expression."""

    def __init__(self, name: str, operator: str, value: any):
        self.name = name
        self.operator = operator
        self.value = value


class TextFilter:
    """Text filter expression."""

    def __init__(self, name: str, operator: str, value: any):
        self.name = name
        self.operator = operator
        self.value = value


# Global constant defining field specifications
VALKEY_LLAMA_FIELD_SPEC = {
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
