from __future__ import annotations

from typing import List, Any, Dict
from uuid import uuid4

from vellum.client import Vellum

from llama_index import Prompt
from llama_index.llm_predictor.vellum.types import VellumDeployment


class VellumPromptRegistry:
    """Registers and retrieves prompts with Vellum.

    A prompt within LlamaIndex should map to a Deployment within Vellum
    at a minimum, and may optionally map to a Sandbox as well.
    """

    def __init__(self, vellum_api_key: str) -> None:
        self._vellum_client = Vellum(api_key=vellum_api_key)

    def from_prompt(self, prompt: Prompt) -> VellumDeployment:
        """Retrieves a prompt from Vellum or registers a new one if needed."""

        deployment_id = prompt.prompt_kwargs.get("vellum_deployment_id")
        deployment_name = prompt.prompt_kwargs.get("vellum_deployment_name")

        if deployment_id or deployment_name:
            return VellumDeployment(
                deployment_id=deployment_id, deployment_name=deployment_name
            )

        registered_prompt = self._register_prompt(prompt)

        return VellumDeployment(
            deployment_id=registered_prompt["deployment"]["id"],
            deployment_name=registered_prompt["deployment"]["name"],
            sandbox_id=registered_prompt["sandbox"]["id"],
        )

    def _register_prompt(self, prompt: Prompt) -> Dict[str, Any]:
        """Registers a prompt with Vellum.

        Prompts should ideally have `vellum_label` and `vellum_name` properties
        within prompt_kwargs set. These are used to identify the entities
        associated with the prompt within Vellum.

        By registering a prompt, Vellum will:
        1) Create a Sandbox for the prompt.
        2) Create a Deployment for the prompt.
        """

        label = (
            prompt.prompt_kwargs.get("vellum_label")
            or f"LlamaIndex Demo: {prompt.prompt_type}"
        )
        name = (
            prompt.prompt_kwargs.get("vellum_name")
            or f"llama-index-demo-{prompt.prompt_type}"
        )

        # TODO: Pass this payload to an API that creates a sandbox,
        #  model version, and deployment.
        payload = {  # noqa
            "name": name,
            "label": label,
            "provider": "OPENAI",
            "model": "text-davinci-003",
            "prompt": self._contruct_prompt_data(prompt),
        }

        return {
            "id": "6f7b18ba-83f4-4fff-baec-56bbbb9ff335",
            "sandbox": {
                "id": "c2fd5533-3932-454c-b5f1-80aa7a7f1491",
                "label": label,
            },
            "model_version": {
                "id": "77aacc88-97cb-4b94-915c-7a90be4edd74",
                "label": label,
            },
            "deployment": {
                "id": "c0f934bd-2e5a-4939-98dc-c772a553a639",
                "name": name,
                "label": label,
            },
        }

    def _contruct_prompt_data(self, prompt: Prompt) -> dict:
        prompt_template = prompt.original_template
        for input_variable in prompt.input_variables:
            prompt_template = prompt_template.replace(
                input_variable, f"{{ {input_variable} }}"
            )

        return {
            "syntax_version": 2,
            "block_data": {
                "version": 1,
                "blocks": [
                    {
                        "id": str(uuid4()),
                        "block_type": "JINJA",
                        "properties": {
                            "template": self._prepare_prompt_jinja_template(
                                prompt.original_template, prompt.input_variables
                            ),
                        },
                    }
                ],
            },
        }

    def _prepare_prompt_jinja_template(
        self, original_template: str, input_variables: List[str]
    ) -> str:
        """Converts a prompt template into a Jinja template."""

        prompt_template = original_template
        for input_variable in input_variables:
            prompt_template = prompt_template.replace(
                input_variable, f"{{ {input_variable} }}"
            )

        return prompt_template
