import time
from typing import Any, Dict, List, Optional
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
    """MyMagicAI LLM.

    Examples:
        `pip install llama-index-llms-mymagic`

        ```python
        from llama_index.llms.mymagic import MyMagicAI

        llm = MyMagicAI(
            api_key="your-api-key",
            storage_provider="s3",  # s3, gcs
            bucket_name="your-bucket-name",
            list_inputs="your list of inputs if you choose to pass directly",
            session="your-session-name",  # files should be located in this folder on which batch inference will be run
            role_arn="your-role-arn",
            system_prompt="your-system-prompt",
            region="your-bucket-region",
            return_output=False,  # Whether you want MyMagic API to return the output json
            input_json_file=None,  # name of the input file (stored on the bucket)
            structured_output=None,  # json schema of the output
            )

        resp = llm.complete(
            question="your-question",
            model="choose-model",  # check models at
            max_tokens=5,  # number of tokens to generate, default is 10
            )

        print(resp)
        ```
    """

    base_url_template: str = "https://fastapi.mymagic.ai"
    completion_url: str = f"{base_url_template}/v1/completions"
    status_url: str = f"{base_url_template}/get_result"

    api_key: str = None
    list_inputs: Optional[List[str]] = Field(
        None,
        description="If user chooses to provide list of inputs to the model instead of specifying in a storage bucket.",
    )
    storage_provider: str = Field(
        default=None, description="The storage provider to use."
    )
    bucket_name: str = Field(
        default=None,
        description="The bucket name where the data is stored.",
    )
    session: str = Field(
        default=None,
        description="The session to use. This is a subfolder in the bucket where your data is located.",
    )
    role_arn: Optional[str] = Field(
        None, description="ARN for role assumption in AWS S3."
    )
    system_prompt: Optional[str] = Field(
        default="Answer the question based only on the given content. Do not give explanations or examples. Do not continue generating more text after the answer.",
        description="The system prompt to use.",
    )
    region: Optional[str] = Field(
        "eu-west-2", description="The region the bucket is in. Only used for AWS S3."
    )

    input_json_file: Optional[str] = Field(
        None, description="Should the input be read from a single json file?"
    )
    structured_output: Optional[Dict[str, Any]] = Field(
        None, description="User-defined structure for the response output"
    )
    model: str = Field(default="mixtral8x7", description="The MyMagicAI model to use.")
    max_tokens: int = Field(
        default=10, description="The maximum number of tokens to generate."
    )
    question = Field(default="", description="The user question.")
    question_data: Dict[str, Any] = Field(
        default_factory=dict, description="The data to send to the MyMagicAI API."
    )
    return_output: Optional[bool] = Field(
        False, description="Whether MyMagic API should return the output json"
    )

    def __init__(
        self,
        api_key: str,
        storage_provider: Optional[str] = None,
        input_json_file: Optional[str] = None,
        structured_output: Optional[Dict[str, Any]] = None,
        return_output: Optional[bool] = False,
        list_inputs: Optional[List[str]] = None,
        role_arn: Optional[str] = None,
        region: Optional[str] = "eu-west-2",
        session: str = None,
        bucket_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.return_output = return_output
        self.api_key = api_key

        self.question_data = {
            "list_inputs": list_inputs,
            "storage_provider": storage_provider,
            "bucket_name": bucket_name,
            "session": session,
            "role_arn": role_arn,
            "system_prompt": system_prompt,
            "region": region,
            "return_output": return_output,
            "input_json_file": input_json_file,
            "structured_output": structured_output,
        }

    @classmethod
    def class_name(cls) -> str:
        return "MyMagicAI"

    async def _submit_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        timeout_config = httpx.Timeout(600.0, connect=60.0)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            resp = await client.post(
                self.completion_url,
                json=question_data,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def _get_result(self, task_id: str) -> Dict[str, Any]:
        url = f"{self.status_url}/{task_id}"
        timeout_config = httpx.Timeout(600.0, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            resp = await client.get(url)
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
        self.question_data["model"] = model or self.model
        self.max_tokens = self.question_data["max_tokens"] = (
            max_tokens or self.max_tokens
        )
        task_response = await self._submit_question(self.question_data)

        if self.return_output:
            return task_response

        task_id = task_response.get("task_id")
        while True:
            result = await self._get_result(task_id)
            if result["status"] != "PENDING":
                return result
            await asyncio.sleep(poll_interval)

    def _submit_question_sync(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submits a question to the model synchronously."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.completion_url,
            json=question_data,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_result_sync(self, task_id: str) -> Dict[str, Any]:
        """Polls for the result of a task synchronously."""
        url = f"{self.status_url}/{task_id}"
        response = requests.get(url, timeout=600.0)
        response.raise_for_status()
        return response.json()

    def complete(
        self,
        question: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        self.question_data["question"] = question
        self.question_data["model"] = model or self.model
        self.max_tokens = self.question_data["max_tokens"] = (
            max_tokens or self.max_tokens
        )
        task_response = self._submit_question_sync(self.question_data)
        if self.return_output:
            return task_response

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
