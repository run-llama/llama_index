from typing import Any, Sequence, List, Optional
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, ValidationError, Extra
from clarifai.client.model import Model

from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    llm_chat_callback,
    llm_completion_callback,
)


class ClarifaiLLM(CustomLLM, extra=Extra.allow):
    """Simple abstract base class for custom LLMs.

    Subclasses must implement the `__init__`, `complete`,
        `stream_complete`, and `metadata` methods.
    """
    
    def __init__(self,
                user_id: Optional[str]= None,
                app_id: Optional[str]= None,
                model_id: Optional[str]= None, 
                *args,
                **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.user_id = user_id
        self.app_id = app_id
        self.model_id = model_id
        # set context window size
        self.context_window = 2048
        # set number of output tokens
        self.num_output = 256
        self.model = Model(user_id=user_id, app_id=app_id, model_id=model_id)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=f"{self.app_id}/{self.model_id}"
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt = bytes(prompt, 'utf-8')
        response = self.model.predict_by_bytes(input_bytes=prompt, input_type="text").outputs[0].data.text.raw
        return CompletionResponse(text=response)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()

    @classmethod
    def class_name(cls) -> str:
        return "custom_llm"
