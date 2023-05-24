from __future__ import annotations

from typing import List, Any
from uuid import uuid4

from llama_index import Prompt
from llama_index.llm_predictor.vellum.types import (
    VellumRegisteredPrompt,
    VellumCompiledPrompt,
)


class VellumPromptRegistry:
    """Registers and retrieves prompts with Vellum.

    LlamaIndex Prompts can be registered within Vellum, at which point Vellum becomes
    the source of truth for the prompt. From there, Vellum can be used for prompt/model
    experimentation, request monitoring, and more.
    """

    def __init__(self, vellum_api_key: str) -> None:
        import_err_msg = (
            "`vellum` package not found, please run `pip install vellum-ai`"
        )
        try:
            from vellum.client import Vellum  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self._vellum_client = Vellum(api_key=vellum_api_key)

    def from_prompt(self, initial_prompt: Prompt) -> VellumRegisteredPrompt:
        """Accepts a LlamaIndex prompt and retrieves a corresponding registered prompt
        from Vellum.

        If the LlamaIndex prompt hasn't yet been registered, it'll be registered
        automatically, after which point Vellum becomes the source-of-truth for the
        prompt's definition.

        In this way, the LlamaIndex prompt is treated as the initial value for the newly
        registered prompt in Vellum.

        You can reference a previously registered prompt by providing either
        `vellum_deployment_id` or `vellum_deployment_name` as keyword arguments
        to `Prompt.prompt_kwargs`.
        """

        deployment_id = initial_prompt.prompt_kwargs.get("vellum_deployment_id")
        deployment_name = initial_prompt.prompt_kwargs.get("vellum_deployment_name")

        registered_prompt: VellumRegisteredPrompt

        if deployment_id or deployment_name:
            deployment_id_or_name: str = (
                deployment_id or deployment_name  # type: ignore
            )
            registered_prompt = self._get_registered_prompt(deployment_id_or_name)
        else:
            registered_prompt = self._register_prompt(initial_prompt)

        return registered_prompt

    def get_compiled_prompt(
        self, registered_prompt: VellumRegisteredPrompt, input_values: dict[str, Any]
    ) -> VellumCompiledPrompt:
        """Retrieves the fully-compiled prompt from Vellum, after all variable
        substitutions, templating, etc.
        """

        result = self._vellum_client.model_versions.model_version_compile_prompt(
            registered_prompt.model_version_id, input_values=input_values
        )
        return VellumCompiledPrompt(
            text=result.prompt.text, num_tokens=result.prompt.num_tokens
        )

    def _get_registered_prompt(
        self, deployment_id_or_name: str
    ) -> VellumRegisteredPrompt:
        """Retrieves a prompt from Vellum, keying off of the deployment's id/name."""

        deployment = self._vellum_client.deployments.retrieve(deployment_id_or_name)

        # Assume that the deployment backing a registered prompt will always have a
        # single model version. Note that this may not be true in the future once
        # deployment-level A/B testing is supported and someone configures an A/B test.
        model_version_id = deployment.active_model_version_ids[0]
        model_version = self._vellum_client.model_versions.retrieve(model_version_id)

        sandbox_snapshot_info = model_version.build_config.sandbox_snapshot
        sandbox_snapshot_id = (
            sandbox_snapshot_info.id if sandbox_snapshot_info else None
        )
        prompt_id = sandbox_snapshot_info.prompt_id if sandbox_snapshot_info else None
        sandbox_id = sandbox_snapshot_info.sandbox_id if sandbox_snapshot_info else None

        return VellumRegisteredPrompt(
            deployment_id=deployment.id,
            deployment_name=deployment.name,
            model_version_id=model_version.id,
            sandbox_id=sandbox_id,
            sandbox_snapshot_id=sandbox_snapshot_id,
            prompt_id=prompt_id,
        )

    def _register_prompt(self, prompt: Prompt) -> VellumRegisteredPrompt:
        """Registers a prompt with Vellum.

        Prompts should ideally have `vellum_label` and `vellum_name` properties
        within prompt_kwargs set. These are used to identify the entities
        associated with the prompt within Vellum.

        By registering a prompt, Vellum will:
        1) Create a Sandbox for the prompt.
        2) Create a Deployment for the prompt.
        """

        # Label represents a human-friendly name that'll be used for all created
        # entities within Vellum. If not provided, a default will be generated.
        label = prompt.prompt_kwargs.get(
            "vellum_label"
        ) or self._generate_default_label(prompt)

        # Name represents a kebab-cased unique identifier that'll be used for all
        # created entities within Vellum. If not provided, a default will be generated.
        name = prompt.prompt_kwargs.get("vellum_name") or self._generate_default_name(
            prompt
        )

        # TODO: Pass this payload to an API that creates a sandbox,
        #  model version, and deployment.
        payload = {  # noqa
            "name": name,
            "label": label,
            "provider": "OPENAI",
            "model": "text-davinci-003",
            "prompt": self._construct_prompt_data(prompt),
        }

        response: dict = {
            "id": "6f7b18ba-83f4-4fff-baec-56bbbb9ff335",
            "sandbox_snapshot": {
                "id": "512f1a46-77e5-4721-8ae0-76a0c2888f29",
                "prompt_id": "6f7b18ba-83f4-4fff-baec-56bbbb9ff335",
            },
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
        return VellumRegisteredPrompt(
            deployment_id=response["deployment"]["id"],
            deployment_name=response["deployment"]["name"],
            model_version_id=response["model_version"]["id"],
            sandbox_id=response["sandbox"]["id"],
            sandbox_snapshot_id=response["sandbox_snapshot"]["id"],
            prompt_id=response["sandbox_snapshot"]["prompt_id"],
        )

    @staticmethod
    def _generate_default_label(prompt: Prompt) -> str:
        return f"LlamaIndex Demo: {prompt.prompt_type}"

    @staticmethod
    def _generate_default_name(prompt: Prompt) -> str:
        return f"llama-index-demo-{prompt.prompt_type}"

    def _construct_prompt_data(
        self, prompt: Prompt, for_chat_model: bool = False
    ) -> dict:
        """Converts a LlamaIndex prompt into Vellum's prompt representation."""

        prompt_template = prompt.original_template
        for input_variable in prompt.input_variables:
            prompt_template = prompt_template.replace(
                input_variable, f"{{ {input_variable} }}"
            )

        block: dict
        jinja_block = {
            "id": str(uuid4()),
            "block_type": "JINJA",
            "properties": {
                "template": self._prepare_prompt_jinja_template(
                    prompt.original_template, prompt.input_variables
                ),
            },
        }
        if for_chat_model:
            block = {
                "id": str(uuid4()),
                "block_type": "CHAT_MESSAGE",
                "properties": {
                    "chat_role": "SYSTEM",
                    "blocks": [jinja_block],
                },
            }
        else:
            block = jinja_block

        return {
            "syntax_version": 2,
            "block_data": {
                "version": 1,
                "blocks": [block],
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
