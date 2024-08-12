"""OpenAI Finetuning."""

import logging
import json
import os
import requests
from typing import Any, Optional

from openai import AzureOpenAI as SyncAzureOpenAI

from llama_index.core.llms.llm import LLM
from llama_index.finetuning.callbacks.finetuning_handler import OpenAIFineTuningHandler
from llama_index.finetuning import OpenAIFinetuneEngine
from llama_index.llms.azure_openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AzureOpenAIFinetuneEngine(OpenAIFinetuneEngine):
    """AzureOpenAI Finetuning Engine."""

    def __init__(
        self,
        base_model: str,
        data_path: str,
        verbose: bool = False,
        start_job_id: Optional[str] = None,
        validate_json: bool = True,
    ) -> None:
        """Init params."""
        self.base_model = base_model
        self.data_path = data_path
        self._verbose = verbose
        self._validate_json = validate_json
        self._start_job: Optional[Any] = None
        self._client = SyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", None),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
        )
        if start_job_id is not None:
            self._start_job = self._client.fine_tuning.jobs.retrieve(start_job_id)

    @classmethod
    def from_finetuning_handler(
        cls,
        finetuning_handler: OpenAIFineTuningHandler,
        base_model: str,
        data_path: str,
        **kwargs: Any,
    ) -> "AzureOpenAIFinetuneEngine":
        """Initialize from finetuning handler.

        Used to finetune an AzureOpenAI model into another
        AzureOpenAI model (e.g. gpt-4o-mini on top of gpt-4o).

        """
        finetuning_handler.save_finetuning_events(data_path)
        return cls(base_model=base_model, data_path=data_path, **kwargs)

    def deploy_finetuned_model(
        self,
        token: str,
        subscription_id: str,
        resource_group: str,
        resource_name: str,
        model_deployment_name: Optional[str] = None,
    ) -> LLM:
        """Deploy finetuned model.

        - token: Azure AD token.
        - subscription_id: 	The subscription ID for the associated Azure OpenAI resource.
        - resource_group: 	The resource group name for your Azure OpenAI resource.
        - resource_name: The Azure OpenAI resource name.
        - model_deployment_name: Custom deployment name that you will use to reference the model when making inference calls.
        """
        current_job = self.get_current_job()

        job_id = current_job.id
        status = current_job.status
        model_id = current_job.fine_tuned_model

        if model_id is None:
            raise ValueError(
                f"Job {job_id} does not have a finetuned model id ready yet."
            )
        if status != "succeeded":
            raise ValueError(f"Job {job_id} has status {status}, cannot get model")

        fine_tuned_model = current_job.fine_tuned_model

        if model_deployment_name is None:
            model_deployment_name = fine_tuned_model

        deploy_params = {"api-version": os.getenv("OPENAI_API_VERSION", "2024-02-01")}
        deploy_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        deploy_data = {
            "sku": {"name": "standard", "capacity": 1},
            "properties": {
                "model": {"format": "OpenAI", "name": fine_tuned_model, "version": "1"}
            },
        }
        deploy_data = json.dumps(deploy_data)
        request_url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}"
        print("Creating a new deployment...")

        response = requests.put(
            request_url, params=deploy_params, headers=deploy_headers, data=deploy_data
        )
        return response.json()

    def get_finetuned_model(self, engine: str, **model_kwargs: Any) -> LLM:
        """Get finetuned model.

        - engine: This will correspond to the custom name you chose
            for your deployment when you deployed a model.
        """
        current_job = self.get_current_job()

        return AzureOpenAI(
            engine=engine or current_job.fine_tuned_model, **model_kwargs
        )
