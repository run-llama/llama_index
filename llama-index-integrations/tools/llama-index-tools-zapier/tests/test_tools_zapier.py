import json
from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.zapier import ZapierToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in ZapierToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def _mock_actions_response(action_id: str, description: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "results": [
                {
                    "id": action_id,
                    "description": f"Zapier: {description}",
                    "params": {},
                }
            ]
        }
    )
    return mock_response


def test_spec_functions_is_not_shared_across_instances() -> None:
    """
    spec_functions used to be a class attribute mutated in place via
    self.spec_functions.append(...) in __init__ - every instance shared
    and appended to the same list, so a second ZapierToolSpec (a
    different API key, a different user) inherited every action name
    the first instance ever registered.
    """
    with patch("llama_index.tools.zapier.base.requests.get") as mock_get:
        mock_get.return_value = _mock_actions_response("1", "Action One")
        tool1 = ZapierToolSpec(api_key="key1")

        mock_get.return_value = _mock_actions_response("2", "Action Two")
        tool2 = ZapierToolSpec(api_key="key2")

    assert tool1.spec_functions == ["Action_One"]
    assert tool2.spec_functions == ["Action_Two"]
