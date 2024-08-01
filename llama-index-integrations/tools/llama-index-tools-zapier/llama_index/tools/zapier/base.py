"""Zapier tool spec."""

import json
from typing import Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

ACTION_URL_TMPL = "https://nla.zapier.com/api/v1/dynamic/exposed/{action_id}/execute/"


class ZapierToolSpec(BaseToolSpec):
    """Zapier tool spec."""

    spec_functions = []

    def __init__(
        self, api_key: Optional[str] = None, oauth_access_token: Optional[str] = None
    ) -> None:
        """Initialize with parameters."""
        if api_key:
            self._headers = {"x-api-key": api_key}
        elif oauth_access_token:
            self._headers = {"Authorization": f"Bearer {oauth_access_token}"}
        else:
            raise ValueError("Must provide either api_key or oauth_access_token")

        # Get the exposed actions from Zapier
        actions = json.loads(self.list_actions())
        if "results" not in actions:
            raise ValueError(
                "No Zapier actions exposed, visit https://nla.zapier.com/dev/actions/"
                " to expose actions."
            )
        results = actions["results"]

        # Register the actions as Tools
        for action in results:
            params = action["params"]

            def function_action(id=action["id"], **kwargs):
                return self.natural_language_query(id, **kwargs)

            action_name = action["description"].split(": ")[1].replace(" ", "_")
            function_action.__name__ = action_name
            function_action.__doc__ = f"""
                This is a Zapier Natural Language Action function wrapper.

                The 'instructions' key is REQUIRED for all function calls.
                The instructions key is a natural language string describing the action to be taken
                The following are all of the valid arguments you can provide: {params}

                Ignore the id field, it is provided for you.
                If the returned error field is not null, interpret the error and try to fix it. Otherwise, inform the user of how they might fix it.
            """
            setattr(self, action_name, function_action)
            self.spec_functions.append(action_name)

    def list_actions(self):
        response = requests.get(
            "https://nla.zapier.com/api/v1/dynamic/exposed/", headers=self._headers
        )
        return response.text

    def natural_language_query(self, id: str, **kwargs):
        response = requests.post(
            ACTION_URL_TMPL.format(action_id=id),
            headers=self._headers,
            data=json.dumps(kwargs),
        )
        return response.text
