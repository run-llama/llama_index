from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from uuid import uuid4

from llama_index.llm_predictor.vellum.types import (
    VellumCompiledPrompt,
    VellumRegisteredPrompt,
)
from llama_index.llm_predictor.vellum.utils import convert_to_kebab_case
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.base import PromptTemplate

if TYPE_CHECKING:
    import vellum


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
            from vellum.client import Vellum
        except ImportError:
            raise ImportError(import_err_msg)

        self._vellum_client = Vellum(api_key=vellum_api_key)

    def from_prompt(self, initial_prompt: BasePromptTemplate) -> VellumRegisteredPrompt:
        """Accepts a LlamaIndex prompt and retrieves a corresponding registered prompt
        from Vellum.

        If the LlamaIndex prompt hasn't yet been registered, it'll be registered
        automatically, after which point Vellum becomes the source-of-truth for the
        prompt's definition.

        In this way, the LlamaIndex prompt is treated as the initial value for the newly
        registered prompt in Vellum.

        You can reference a previously registered prompt by providing either
        `vellum_deployment_id` or `vellum_deployment_name` as key/value pairs within
        `BasePromptTemplate.metadata`.
        """
        from vellum.core import ApiError

        deployment_id = initial_prompt.metadata.get("vellum_deployment_id")
        deployment_name = initial_prompt.metadata.get(
            "vellum_deployment_name"
        ) or self._generate_default_name(initial_prompt)

        registered_prompt: VellumRegisteredPrompt
        try:
            deployment = self._vellum_client.deployments.retrieve(
                deployment_id or deployment_name
            )
        except ApiError as e:
            if e.status_code == 404:
                registered_prompt = self._register_prompt(initial_prompt)
            else:
                raise
        else:
            registered_prompt = self._get_registered_prompt(deployment)

        return registered_prompt

    def get_compiled_prompt(
        self, registered_prompt: VellumRegisteredPrompt, input_values: Dict[str, Any]
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
        self, deployment: vellum.DeploymentRead
    ) -> VellumRegisteredPrompt:
        """Retrieves a prompt from Vellum, keying off of the deployment's id/name."""
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

    def _register_prompt(self, prompt: BasePromptTemplate) -> VellumRegisteredPrompt:
        """Registers a prompt with Vellum.

        By registering a prompt, Vellum will:
        1) Create a Sandbox for the prompt so that you can experiment with the
              prompt, LLM provider, model, and parameters via Vellum's UI.
        2) Deployment for the prompt so that you can monitor requests and
            update the prompt, LLM provider, model, and parameters via Vellum's UI
            without requiring code changes.
        """
        # Label represents a human-friendly name that'll be used for all created
        # entities within Vellum. If not provided, a default will be generated.
        label = prompt.metadata.get(
            "vellum_deployment_label"
        ) or self._generate_default_label(prompt)

        # Name represents a kebab-cased unique identifier that'll be used for all
        # created entities within Vellum. If not provided, a default will be generated.
        name = prompt.metadata.get(
            "vellum_deployment_name"
        ) or self._generate_default_name(prompt)

        # Note: For now, the initial provider, model, and parameters used to register
        # the prompt are hard-coded. You can then update any of these from within
        # Vellum's UI. As a future improvement, we could allow these to be specified
        # upfront.
        provider, model, params = self._get_default_llm_meta()
        prompt_info = self._construct_prompt_info(prompt, for_chat_model=True)

        resp = self._vellum_client.registered_prompts.register_prompt(
            label=label,
            name=name,
            prompt=prompt_info,
            provider=provider,
            model=model,
            parameters=params,
            meta={
                "source": "llamaindex",
                "prompt_type": prompt.metadata["prompt_type"],
            },
        )

        return VellumRegisteredPrompt(
            deployment_id=resp.deployment.id,
            deployment_name=resp.deployment.name,
            model_version_id=resp.model_version.id,
            sandbox_id=resp.sandbox.id,
            sandbox_snapshot_id=resp.sandbox_snapshot.id,
            prompt_id=resp.prompt.id,
        )

    def _generate_default_label(self, prompt: BasePromptTemplate) -> str:
        prompt_type = prompt.metadata["prompt_type"]
        return f"LlamaIndex Demo: {prompt_type}'"

    def _generate_default_name(self, prompt: BasePromptTemplate) -> str:
        default_label = self._generate_default_label(prompt)
        return convert_to_kebab_case(default_label)

    def _construct_prompt_info(
        self, prompt: BasePromptTemplate, for_chat_model: bool = True
    ) -> vellum.RegisterPromptPromptInfoRequest:
        """Converts a LlamaIndex prompt into Vellum's prompt representation."""
        import vellum

        assert isinstance(prompt, PromptTemplate)
        prompt_template = prompt.template
        for input_variable in prompt.template_vars:
            prompt_template = prompt_template.replace(
                input_variable, f"{{ {input_variable} }}"
            )

        block: vellum.PromptTemplateBlockRequest
        jinja_block = vellum.PromptTemplateBlockRequest(
            id=str(uuid4()),
            block_type=vellum.BlockTypeEnum.JINJA,
            properties=vellum.PromptTemplateBlockPropertiesRequest(
                template=self._prepare_prompt_jinja_template(
                    prompt.template,
                    prompt.template_vars,
                ),
            ),
        )
        if for_chat_model:
            block = vellum.PromptTemplateBlockRequest(
                id=str(uuid4()),
                block_type=vellum.BlockTypeEnum.CHAT_MESSAGE,
                properties=vellum.PromptTemplateBlockPropertiesRequest(
                    chat_role=vellum.ChatMessageRole.SYSTEM,
                    blocks=[jinja_block],
                ),
            )
        else:
            block = jinja_block

        return vellum.RegisterPromptPromptInfoRequest(
            prompt_syntax_version=2,
            prompt_block_data=vellum.PromptTemplateBlockDataRequest(
                version=1,
                blocks=[block],
            ),
            input_variables=[{"key": input_var} for input_var in prompt.template_vars],
        )

    def _prepare_prompt_jinja_template(
        self, original_template: str, input_variables: List[str]
    ) -> str:
        """Converts a prompt template into a Jinja template."""
        prompt_template = original_template
        for input_variable in input_variables:
            prompt_template = prompt_template.replace(
                ("{" + input_variable + "}"), ("{{ " + input_variable + " }}")
            )

        return prompt_template

    def _get_default_llm_meta(
        self,
    ) -> Tuple[vellum.ProviderEnum, str, vellum.RegisterPromptModelParametersRequest]:
        import vellum

        return (
            vellum.ProviderEnum.OPENAI,
            "gpt-3.5-turbo",
            vellum.RegisterPromptModelParametersRequest(
                temperature=0.0,
                max_tokens=256,
                stop=[],
                top_p=1.0,
                top_k=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logit_bias=None,
            ),
        )
