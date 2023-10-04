from typing import Any, Sequence, List, Optional
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, ValidationError, Extra
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    LLM,
    CompletionResponse,
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
                additional_kwargs: Optional[Dict[str, Any]] = None,
                callback_manager: Optional[CallbackManager] = None,
    )-> None :
        try:
            from clarifai.client.model import Model
        except:
            raise ImportError("ClarifaiEmbedding requires the Clarifai package to be installed.\nPlease install the package with `pip install clarifai`.")
        
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
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
        try:
            response = self.model.predict_by_bytes(input_bytes=prompt, input_type="text").outputs[0].data.text.raw
        except Exception as e:
            print(f"Model prediction failed {e}")

        return CompletionResponse(text=response)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()

    @classmethod
    def class_name(cls) -> str:
        return "custom_llm"
