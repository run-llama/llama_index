from typing import Optional, List
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.schema import Document
from hive_intelligence.client import HiveSearchClient
from hive_intelligence.types import HiveSearchRequest, HiveSearchMessage, HiveSearchResponse
from hive_intelligence.errors import HiveSearchAPIError

class HiveToolSpec(BaseToolSpec):
    """Hive Search tool spec."""

    spec_functions = ["search"]

    def __init__(self, api_key: str) -> None:
        self.client = HiveSearchClient(api_key=api_key)

    def search(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[HiveSearchMessage]] = None,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_data_sources: bool = False
    ) -> HiveSearchResponse:
        """
        Executes a Hive search request via prompt or chat-style messages.
        """
        req = HiveSearchRequest(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            include_data_sources=include_data_sources,
        )
        try:
            response= self.client.search(req)
        except HiveSearchAPIError as e:
            raise RuntimeError(f"{e}") from e

        # Wrap each returned result into Llama‑Index Document
        return response
        # for item in response.results:  # assuming .results list in response model
        #     text = item.content or item.answer
        #     extra = {}
        #     if include_data_sources and item.data_sources:
        #         extra["sources"] = item.data_sources
        #     docs.append(Document(text=text, extra_info=extra))
        # return docs





# """Hive Tool Spec."""

# import json
# import logging
# from typing import List, Optional, Dict, Any, Union, Literal, TypedDict, overload
# import os

# from llama_index.core.tools.tool_spec.base import BaseToolSpec
# from llama_index.core.schema import Document

# logger = logging.getLogger(__name__)

# class HiveMessage(TypedDict):
#     """A message in the chat format expected by Hive Intelligence."""
#     role: Literal["user", "assistant", "system"]
#     content: str

# class HiveToolSpec(BaseToolSpec):
#     """Hive Tool Spec.
    
#     This tool provides access to Hive Intelligence's search and chat capabilities.
#     It requires an API key from Hive Intelligence.
#     """

#     spec_functions = ["search"]

#     def __init__(self, api_key: Optional[str] = None):
#         """Initialize with parameters.
        
#         Args:
#             api_key (str, optional): Hive API key. Defaults to None, in which case
#                 it will be read from the HIVE_API_KEY environment variable.
#         """
#         try:
#             from hive_intelligence import (
#                 HiveSearchClient, 
#                 HiveSearchRequest, 
#                 HiveSearchMessage,
#                 HiveSearchResponse,
#                 HiveSearchAPIError
#             )
#             self.HiveSearchMessage = HiveSearchMessage
#             self.HiveSearchResponse = HiveSearchResponse
#             self.HiveSearchAPIError = HiveSearchAPIError
#         except ImportError as e:
#             raise ImportError(
#                 "Please install the hive-intelligence package: "
#                 "`pip install hive-intelligence`"
#             ) from e
            
#         self.HiveSearchClient = HiveSearchClient
#         self.HiveSearchRequest = HiveSearchRequest
        
#         self.api_key = api_key or os.getenv("HIVE_API_KEY")
#         if not self.api_key:
#             raise ValueError(
#                 "Please provide an API key either through the api_key parameter "
#                 "or the HIVE_API_KEY environment variable."
#             )
            
#         self.client = HiveSearchClient(api_key=self.api_key)

#     def _make_api_request(
#         self,
#         request_params: Dict[str, Any],
#     ) -> List[Document]:
#         """Make an API request to Hive Intelligence.
        
#         Args:
#             request_params: Parameters for the API request.
            
#         Returns:
#             List[Document]: Response from the API.
#         """
#         try:
#             # Filter out None values
#             request_params = {k: v for k, v in request_params.items() if v is not None}
            
#             # Create and send request
#             request = self.HiveSearchRequest(**request_params)
#             logger.debug(f"Created request: {request}")
            
#             try:
#                 response = self.client.search(request)
#                 logger.debug(f"Received response: {response}")
                
#                 # Extract response data
#                 response_data = getattr(response, 'response', {})
#                 data_sources = getattr(response, 'data_sources', [])
#                 is_additional_data_required = getattr(response, 'isAdditionalDataRequired', None)
                
#                 # Convert response to a readable string
#                 try:
#                     if isinstance(response_data, (str, int, float, bool)):
#                         response_text = str(response_data)
#                     else:
#                         response_text = json.dumps(response_data, indent=2)
#                 except (TypeError, ValueError) as e:
#                     logger.warning(f"Could not serialize response: {e}")
#                     response_text = str(response_data)
                
#                 # Prepare metadata
#                 metadata = {
#                     "model": "hive-intelligence",
#                     "data_sources": data_sources,
#                     "isAdditionalDataRequired": is_additional_data_required,
#                     **{k: v for k, v in request_params.items() 
#                        if k not in ['messages', 'prompt']}
#                 }
                
#                 # Include raw response in metadata for debugging
#                 try:
#                     metadata["raw_response"] = response.model_dump()
#                 except Exception as e:
#                     logger.debug(f"Could not get raw response: {e}")
#                     metadata["raw_response"] = str(response)
                    
#                 return [Document(text=response_text, metadata=metadata)]
                
#             except self.HiveSearchAPIError as e:
#                 logger.error(f"HiveSearch API Error: {e.status_code} {e.reason} - {e.message}")
#                 return [Document(
#                     text=f"Hive Intelligence API Error ({e.status_code}): {e.message}",
#                     metadata={
#                         "error": True,
#                         "error_type": "api_error",
#                         "status_code": e.status_code,
#                         "reason": e.reason,
#                         "message": e.message
#                     }
#                 )]
                
#         except Exception as e:
#             logger.error(f"Unexpected error calling HiveSearch API: {str(e)}")
#             logger.error(f"Exception type: {type(e).__name__}")
#             logger.error(f"Exception args: {e.args}")
            
#             error_details = {
#                 "error": True,
#                 "error_type": "unexpected_error",
#                 "exception_type": type(e).__name__,
#                 "exception_args": str(e.args) if e.args else "No args"
#             }
            
#             if hasattr(e, 'response'):
#                 error_details["http_status"] = getattr(e.response, 'status_code', None)
#                 error_details["http_reason"] = getattr(e.response, 'reason', None)
            
#             return [Document(
#                 text=f"Error querying Hive Intelligence: {str(e)} (Type: {type(e).__name__})",
#                 metadata=error_details
#             )]

#     @overload
#     def search(
#         self,
#         prompt: str,
#         temperature: float = 0.7,
#         top_k: Optional[int] = None,
#         top_p: Optional[float] = None,
#         max_tokens: Optional[int] = None,
#         include_data_sources: bool = True,
#     ) -> List[Document]:
#         ...

#     @overload
#     def search(
#         self,
#         messages: List[Union[Dict[str, str], HiveMessage]],
#         temperature: float = 0.7,
#         top_k: Optional[int] = None,
#         top_p: Optional[float] = None,
#         max_tokens: Optional[int] = None,
#         include_data_sources: bool = True,
#     ) -> List[Document]:
#         ...

#     def search(
#         self,
#         prompt: Optional[str] = None,
#         messages: Optional[List[Union[Dict[str, str], HiveMessage]]] = None,
#         temperature: float = 0.7,
#         top_k: Optional[int] = None,
#         top_p: Optional[float] = None,
#         max_tokens: Optional[int] = None,
#         include_data_sources: bool = True,
#     ) -> List[Document]:
#         """Search or chat with Hive Intelligence.
        
#         Either `prompt` or `messages` must be provided, but not both.
        
#         Args:
#             prompt: A single search query or prompt. Mutually exclusive with `messages`.
#             messages: List of message dictionaries with 'role' and 'content' keys for 
#                     chat-style interactions. Mutually exclusive with `prompt`.
#             temperature: Controls randomness (0.0 to 1.0).
#             top_k: Limits the number of highest probability vocabulary tokens to keep.
#             top_p: Nucleus sampling parameter for controlling diversity.
#             max_tokens: Maximum number of tokens in the response.
#             include_data_sources: Whether to include data sources in the response.
            
#         Returns:
#             List[Document]: A list containing a Document with the response and metadata.
            
#         Raises:
#             ValueError: If neither or both of prompt/messages are provided, or if messages 
#                      format is invalid.
#         """
#         # Validate that exactly one of prompt or messages is provided
#         if prompt is not None and messages is not None:
#             raise ValueError("Only one of 'prompt' or 'messages' should be provided, not both.")
#         if prompt is None and messages is None:
#             raise ValueError("Either 'prompt' or 'messages' must be provided.")
            
#         request_params = {
#             "temperature": temperature,
#             "include_data_sources": include_data_sources,
#             "top_k": top_k,
#             "top_p": top_p,
#             "max_tokens": max_tokens,
#         }
        
#         if prompt is not None:
#             if not isinstance(prompt, str) or not prompt.strip():
#                 raise ValueError("Prompt must be a non-empty string")
#             request_params["prompt"] = prompt
#             logger.debug(f"Sending prompt request: {prompt[:100]}...")
#         else:  # messages is not None
#             if not isinstance(messages, list) or not messages:
#                 raise ValueError("Messages must be a non-empty list of message dictionaries")
                
#             hive_messages = []
#             for i, msg in enumerate(messages, 1):
#                 if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
#                     raise ValueError(
#                         f"Message {i} is invalid. Each message must be a dictionary with 'role' and 'content' keys. "
#                         f"Got: {msg}"
#                     )
#                 hive_messages.append(
#                     self.HiveSearchMessage(role=msg["role"], content=str(msg["content"]))
#                 )
            
#             request_params["messages"] = hive_messages
#             logger.debug(f"Sending chat request with {len(hive_messages)} messages")
        
#         return self._make_api_request(request_params)