from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.google import (
    GmailToolSpec,
    GoogleCalendarToolSpec,
    GoogleSearchToolSpec,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in GoogleCalendarToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GmailToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GoogleSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes
