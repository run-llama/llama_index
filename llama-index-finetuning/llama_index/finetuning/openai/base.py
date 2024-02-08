"""OpenAI Finetuning."""

import logging
import os
import time
from typing import Any, Optional

import openai
from openai import OpenAI as SyncOpenAI
from openai.types.fine_tuning import FineTuningJob

from llama_index.core.llms.llm import LLM
from llama_index.finetuning.callbacks.finetuning_handler import OpenAIFineTuningHandler
from llama_index.finetuning.openai.validate_json import validate_json
from llama_index.finetuning.types import BaseLLMFinetuneEngine
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIFinetuneEngine(BaseLLMFinetuneEngine):
    """OpenAI Finetuning Engine."""

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
        self._client = SyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", None))
        if start_job_id is not None:
            self._start_job = self._client.fine_tuning.jobs.retrieve(start_job_id)

    @classmethod
    def from_finetuning_handler(
        cls,
        finetuning_handler: OpenAIFineTuningHandler,
        base_model: str,
        data_path: str,
        **kwargs: Any,
    ) -> "OpenAIFinetuneEngine":
        """Initialize from finetuning handler.

        Used to finetune an OpenAI model into another
        OpenAI model (e.g. gpt-3.5-turbo on top of GPT-4).

        """
        finetuning_handler.save_finetuning_events(data_path)
        return cls(base_model=base_model, data_path=data_path, **kwargs)

    def finetune(self) -> None:
        """Finetune model."""
        if self._validate_json:
            validate_json(self.data_path)

        # TODO: figure out how to specify file name in the new API
        # file_name = os.path.basename(self.data_path)

        # upload file
        with open(self.data_path, "rb") as f:
            output = self._client.files.create(file=f, purpose="fine-tune")
        logger.info("File uploaded...")
        if self._verbose:
            print("File uploaded...")

        # launch training
        while True:
            try:
                job_output = self._client.fine_tuning.jobs.create(
                    training_file=output.id, model=self.base_model
                )
                self._start_job = job_output
                break
            except openai.BadRequestError:
                print("Waiting for file to be ready...")
                time.sleep(60)
        info_str = (
            f"Training job {output.id} launched. "
            "You will be emailed when it's complete."
        )
        logger.info(info_str)
        if self._verbose:
            print(info_str)

    def get_current_job(self) -> FineTuningJob:
        """Get current job."""
        # validate that it works
        if not self._start_job:
            raise ValueError("Must call finetune() first")

        # try getting id, make sure that run succeeded
        job_id = self._start_job.id
        return self._client.fine_tuning.jobs.retrieve(job_id)

    def get_finetuned_model(self, **model_kwargs: Any) -> LLM:
        """Gets finetuned model."""
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

        return OpenAI(model=model_id, **model_kwargs)
