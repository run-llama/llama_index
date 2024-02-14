import time
from typing import Any, Dict
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
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import LLM
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class MyMagicAI(LLM):
    base_url_template: str = "https://{model}.mymagic.ai"
    default_model: str = "mistral7b"
    DEFAULT_NUM_OUTPUTS: int = 10
    model: str = "mistral7b"    
    api_key: str = "test______________________Vitai"
    
    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key


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
            resp = await client.get(f"{self._construct_url(self.model)}/get_result/{task_id}")
            resp.raise_for_status()
            return resp.json()

    async def acomplete(self, question: str, poll_interval: float = 1.0) -> CompletionResponse:
        question_data = {
            "question": question,
            "personal_access_token": self.api_key,
            "storage_provider": "gcs",
            "bucket_name": "vitali-mymagic", 
            "session": "test-session", 
            "max_tokens": 20
        }
        task_response = await self._submit_question(question_data)
        task_id = task_response.get("task_id")
        while True:
            result = await self._get_result(task_id)
            if result["status"] != "PENDING":
                return result
            await asyncio.sleep(poll_interval)


    def complete(self, question: str, poll_interval: float = 1.0) -> CompletionResponse:
        """Submits a question synchronously and polls for the result."""
        question_data = {
            "question": question,
            "personal_access_token": self.api_key,
            "storage_provider": "gcs",
            "bucket_name": "vitali-mymagic", 
            "session": "test-session", 
            "max_tokens": 20
        }
        task_response = self._submit_question_sync(question_data)
        task_id = task_response.get("task_id")
        while True:
            result = self._get_result_sync(task_id)
            if result["status"] != "PENDING":
                # Process and return the result as a CompletionResponse
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
        raise NotImplementedError(
            "MyMagicAI does not currently support chat."
        )       
    def chat(self, question: str) -> ChatResponse:
        raise NotImplementedError(
            "MyMagicAI does not currently support chat."
        )    
    async def astream_complete(self, question: str) -> CompletionResponseAsyncGen:
        raise NotImplementedError(
            "MyMagicAI does not currently support streaming."
        )    
    async def astream_chat(self, question: str) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "MyMagicAI does not currently support streaming."
        )
    def chat(self, question: str) -> ChatResponse:
        raise NotImplementedError(
            "MyMagicAI does not currently support chat."
        )    
    def stream_chat(self, question: str) -> ChatResponseGen:
        raise NotImplementedError(
            "MyMagicAI does not currently support chat."
        ) 
                
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=False,
        )