"""MistralAI Finetuning."""

import logging
import os
import time
from typing import Any, Optional, Dict
import sys

from mistralai import Mistral
from mistralai.models import (
    JobsAPIRoutesFineTuningGetFineTuningJobResponse,
    WandbIntegration,
    CompletionTrainingParametersIn,
)

from llama_index.core.llms.llm import LLM
from llama_index.finetuning.callbacks.finetuning_handler import (
    MistralAIFineTuningHandler,
)
from llama_index.finetuning.mistralai.utils import reformat_jsonl
from llama_index.finetuning.types import BaseLLMFinetuneEngine
from llama_index.llms.mistralai import MistralAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)


class MistralAIFinetuneEngine(BaseLLMFinetuneEngine):
    """MistralAI Finetuning Engine."""

    def __init__(
        self,
        base_model: str,
        training_path: str,
        validation_path: Optional[str] = None,
        verbose: bool = False,
        start_job_id: Optional[str] = None,
        validate_json: bool = True,
        training_steps: int = 10,
        learning_rate: float = 0.0001,
        wandb_integration_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        """Init params."""
        self.base_model = base_model
        self.training_path = training_path
        self.validation_path = validation_path
        self._verbose = verbose
        self._validate_json = validate_json
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.wandb_integration_dict = wandb_integration_dict
        self._client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", None))
        self._start_job: Optional[Any] = None
        if start_job_id is not None:
            self._start_job = self._client.fine_tuning.jobs.get(start_job_id)

    @classmethod
    def from_finetuning_handler(
        cls,
        finetuning_handler: MistralAIFineTuningHandler,
        base_model: str,
        training_path: str,
        **kwargs: Any,
    ) -> "MistralAIFinetuneEngine":
        """
        Initialize from finetuning handler.

        Used to finetune an MistralAI model.

        """
        finetuning_handler.save_finetuning_events(training_path)
        return cls(base_model=base_model, data_path=training_path, **kwargs)

    def finetune(self) -> None:
        """Finetune model."""
        if self._validate_json:
            if self.training_path:
                reformat_jsonl(self.training_path)
            if self.validation_path:
                reformat_jsonl(self.validation_path)

        # upload file
        with open(self.training_path, "rb") as f:
            train_file = self._client.files.upload(file=f)
        if self.validation_path:
            with open(self.validation_path, "rb") as f:
                eval_file = self._client.files.upload(file=f)
        logger.info("File uploaded...")
        if self._verbose:
            print("File uploaded...")

        # launch training
        while True:
            try:
                job_output = self._client.fine_tuning.jobs.create(
                    training_files=[train_file.id],
                    validation_files=[eval_file.id] if self.validation_path else None,
                    model=self.base_model,
                    hyperparameters=CompletionTrainingParametersIn(
                        training_steps=self.training_steps,
                        learning_rate=self.learning_rate,
                    ),
                    integrations=(
                        [
                            WandbIntegration(
                                project=self.wandb_integration_dict["project"],
                                run_name=self.wandb_integration_dict["run_name"],
                                api_key=self.wandb_integration_dict["api_key"],
                            ).model_dump()
                        ]
                        if self.wandb_integration_dict
                        else None
                    ),
                )
                self._start_job = job_output
                break
            except Exception:
                print("Waiting for file to be ready...")
                time.sleep(60)
        info_str = f"Training job {job_output.id} launched. "
        logger.info(info_str)
        if self._verbose:
            print(info_str)

    def get_current_job(
        self,
    ) -> Optional[JobsAPIRoutesFineTuningGetFineTuningJobResponse]:
        """Get current job."""
        # validate that it works
        if not self._start_job:
            raise ValueError("Must call finetune() first")

        # try getting id, make sure that run succeeded
        job_id = self._start_job.id
        return self._client.fine_tuning.jobs.get(job_id)

    def get_finetuned_model(self, **model_kwargs: Any) -> LLM:
        """Gets finetuned model."""
        current_job = self.get_current_job()

        job_id = current_job.id
        status = current_job.status
        model_id = current_job.fine_tuned_model

        logger.info(f"status of the job_id: {job_id} is {status}")

        if model_id is None:
            raise ValueError(
                f"Job {job_id} does not have a finetuned model id ready yet."
            )
        if status != "SUCCESS":
            raise ValueError(f"Job {job_id} has status {status}, cannot get model")

        return MistralAI(model=model_id, **model_kwargs)
