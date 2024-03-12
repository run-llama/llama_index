import time
from typing import Any, Dict, Optional
import httpx
import asyncio
import requests
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.bridge.pydantic import Field


class MyMagicAI(LLM):
    base_url_template: str = "https://{model}.mymagic.ai"
    api_key: str = None
    model: str = Field(default="mistral7b", description="The MyMagicAI model to use.")
    max_tokens: int = Field(
        default=10, description="The maximum number of tokens to generate."
    )
    question = Field(default="", description="The user question.")
    storage_provider: str = Field(
        default="gcs", description="The storage provider to use."
    )
    bucket_name: str = Field(
        default="your-bucket-name",
        description="The bucket name where the data is stored.",
    )
    session: str = Field(
        default="test-session",
        description="The session to use. This is a subfolder in the bucket where your data is located.",
    )
    role_arn: Optional[str] = Field(
        None, description="ARN for role assumption in AWS S3."
    )
    system_prompt: str = Field(
        default="Answer the question based only on the given content. Do not give explanations or examples. Do not continue generating more text after the answer.",
        description="The system prompt to use.",
    )
    question_data: Dict[str, Any] = Field(
        default_factory=dict, description="The data to send to the MyMagicAI API."
    )
    region: Optional[str] = Field(
        "eu-west-2", description="The region the bucket is in. Only used for AWS S3."
    )
    return_output: Optional[bool] = Field(
        False, description="Whether MyMagic API should return the output json"
    )

    def __init__(
        self,
        api_key: str,
        storage_provider: str,
        bucket_name: str,
        session: str,
        system_prompt: Optional[str],
        role_arn: Optional[str] = None,
        region: Optional[str] = None,
        return_output: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.question_data = {
            "storage_provider": storage_provider,
            "bucket_name": bucket_name,
            "personal_access_token": api_key,
            "session": session,
            "max_tokens": self.max_tokens,
            "role_arn": role_arn,
            "system_prompt": system_prompt,
            "region": region,
            "return_output": return_output,
        }

    @classmethod
    def class_name(cls) -> str:
        return "MyMagicAI"

    def _construct_url(self, model: str) -> str:
        """Constructs the API endpoint URL based on the specified model."""
        return self.base_url_template.format(model=model)

    async def _submit_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            url = f"{self._construct_url(self.model)}/submit_question"
            resp = await client.post(url, json=question_data)
            resp.raise_for_status()
            return resp.json()

    def _submit_question_sync(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submits a question to the model synchronously."""
        url = f"{self._construct_url(self.model)}/submit_question"
        resp = requests.post(url, json=question_data)
        resp.raise_for_status()
        return resp.json()

    def _get_result_sync(self, task_id: str) -> Dict[str, Any]:
        """Polls for the result of a task synchronously."""
        url = f"{self._construct_url(self.model)}/get_result/{task_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    async def _get_result(self, task_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._construct_url(self.model)}/get_result/{task_id}"
            )
            resp.raise_for_status()
            return resp.json()

    async def acomplete(
        self,
        question: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        self.question_data["question"] = question
        self.model = self.question_data["model"] = model or self.model
        self.max_tokens = self.question_data["max_tokens"] = (
            max_tokens or self.max_tokens
        )

        task_response = await self._submit_question(self.question_data)
        task_id = task_response.get("task_id")
        while True:
            result = await self._get_result(task_id)
            if result["status"] != "PENDING":
                return result
            await asyncio.sleep(poll_interval)

    def complete(
        self,
        question: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        self.question_data["question"] = question
        self.model = self.question_data["model"] = model or self.model
        self.max_tokens = self.question_data["max_tokens"] = (
            max_tokens or self.max_tokens
        )

        task_response = self._submit_question_sync(self.question_data)
        task_id = task_response.get("task_id")
        while True:
            result = self._get_result_sync(task_id)
            if result["status"] != "PENDING":
                return CompletionResponse(
                    text=result.get("message", ""),
                    additional_kwargs={"status": result["status"]},
                )
            time.sleep(poll_interval)

    def stream_complete(self, question: str) -> CompletionResponseGen:
        raise NotImplementedError(
            "MyMagicAI does not currently support streaming completion."
        )

    async def achat(self, question: str) -> ChatResponse:
        raise NotImplementedError("MyMagicAI does not currently support chat.")

    def chat(self, question: str) -> ChatResponse:
        raise NotImplementedError("MyMagicAI does not currently support chat.")

    async def astream_complete(self, question: str) -> CompletionResponseAsyncGen:
        raise NotImplementedError("MyMagicAI does not currently support streaming.")

    async def astream_chat(self, question: str) -> ChatResponseAsyncGen:
        raise NotImplementedError("MyMagicAI does not currently support streaming.")

    def chat(self, question: str) -> ChatResponse:
        raise NotImplementedError("MyMagicAI does not currently support chat.")

    def stream_chat(self, question: str) -> ChatResponseGen:
        raise NotImplementedError("MyMagicAI does not currently support chat.")

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=False,
        )
