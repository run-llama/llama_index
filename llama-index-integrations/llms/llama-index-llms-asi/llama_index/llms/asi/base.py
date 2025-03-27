"""ASI LLM implementation."""

import os
from typing import Any, Optional, Sequence, Iterator, Dict, List, cast

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai.base import (
    ChatMessage,
    ChatResponseGen,
    CompletionResponseGen,
    CompletionResponse,
    ChatResponse,
    MessageRole,
)

DEFAULT_MODEL = "asi1-mini"


class ASI(OpenAILike):
    """
    ASI LLM - Integration for ASI models.
    
    Currently supported models:
    - asi1-mini
    
    Examples:
        `pip install llama-index-llms-asi`

        ```python
        from llama_index.llms.asi import ASI

        # Set up the ASI class with the required model and API key
        llm = ASI(model="asi1-mini", api_key="your_api_key")
        
        # Call the complete method with a query
        response = llm.complete("Explain the importance of AI")
        
        print(response)
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: str = "https://api.asi1.ai/v1",
        is_chat_model: bool = True,
        is_function_calling_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ASI LLM.

        Args:
            model (str): The ASI model to use. Defaults to "asi1-mini".
            api_key (Optional[str]): The API key to use. If None, the ASI_API_KEY
                environment variable will be used. Defaults to None.
            api_base (str): The base URL for the ASI API. Defaults to
                "https://api.asi1.ai/v1".
            is_chat_model (bool): Whether the model supports chat. Defaults to True.
            is_function_calling_model (bool): Whether the model supports function
                calling. Defaults to False.
            **kwargs (Any): Additional arguments to pass to the OpenAILike constructor.
        """
        api_key = api_key or os.environ.get("ASI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "ASI API key is required. Set it using the api_key parameter "
                "or the ASI_API_KEY environment variable."
            )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )
    
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Override stream_complete to handle ASI's limitations.
        
        ASI doesn't support the completions endpoint at all (returns 404 error),
        so we use a fallback mechanism that returns the complete response as a single chunk.
        """
        # Get a complete response using the non-streaming complete method
        response = self.complete(prompt, formatted=formatted, **kwargs)
        
        # Create a single chunk with the complete response
        if response and response.text:
            # Create a copy of the response to avoid modifying the original
            chunk = response
            
            # Yield the chunk
            yield chunk
    
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Override stream_chat to handle ASI's unique streaming format.
        
        ASI's streaming format includes custom fields like 'thought' and 'init_thought'
        that aren't part of the standard OpenAI format. This method processes the
        raw stream to extract meaningful content from these fields if available.
        """
        # Call the parent's stream_chat method to get the raw stream
        raw_stream = super().stream_chat(messages, **kwargs)
        
        # Process the stream to handle ASI's unique format
        accumulated_content = ""
        has_yielded_content = False
        
        for chunk in raw_stream:
            # Check if this chunk has content in the standard delta.content field
            if hasattr(chunk, 'raw') and hasattr(chunk.raw, 'choices'):
                for choice in chunk.raw.choices:
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        content = choice.delta.content
                        if content and content.strip():
                            # Update the chunk's message content with the actual content
                            if chunk.message:
                                chunk.message.content = content
                            else:
                                # Create a new message if needed
                                chunk.message = ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=content
                                )
                            accumulated_content += content
                            has_yielded_content = True
                            yield chunk
            
            # If this is the final chunk with usage info and we haven't yielded anything
            if not has_yielded_content and hasattr(chunk, 'additional_kwargs') and \
               'prompt_tokens' in chunk.additional_kwargs:
                # Create a synthetic response with the accumulated content if available
                if accumulated_content:
                    # Create a new message with the accumulated content
                    message = ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=accumulated_content
                    )
                    # Create a new ChatResponse with the message
                    response = ChatResponse(message=message, raw=chunk.raw)
                    yield response
                else:
                    # Fallback to non-streaming if we couldn't extract any content
                    response = self.chat(messages, **kwargs)
                    yield response

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ASI"