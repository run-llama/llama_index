import os
from typing import Any, Dict, Optional, List

import requests
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.bridge.pydantic import Field, PrivateAttr
import json


class Sambaverse(CustomLLM):
    """
    Sambaverse LLM.

    Examples:
        `pip install llama-index-llms-sambanova`

        ```python
        from llama_index.llms.sambanova import Sambaverse

        llm = Sambaverse(...)

        response = llm.complete("What is the meaning of life?")

        print(response)
        ```
    """

    sambaverse_url: str = Field(
        default="https://sambaverse.sambanova.ai",
        description="URL of the Sambaverse server",
    )

    sambaverse_api_key: str = Field(
        default="",
        description="API key for the Sambaverse server",
    )
    sambaverse_model_name: str = Field(
        default="", description="Name of the Sambaverse model to use"
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the model",
    )
    streaming: bool = Field(
        default=False,
        description="Boolean to state whether to stream response or not",
    )

    _client: requests.Session = PrivateAttr()

    def __init__(
        self,
        sambaverse_url: str = "https://sambaverse.sambanova.ai",
        sambaverse_api_key: str = "",
        sambaverse_model_name: str = "",
        model_kwargs: Dict[str, Any] = {},
        streaming: bool = False,
        client: Optional[requests.Session] = None,
    ) -> None:
        super().__init__()

        self.sambaverse_url = sambaverse_url
        self.sambaverse_api_key = sambaverse_api_key
        self.sambaverse_model_name = sambaverse_model_name
        self.model_kwargs = model_kwargs
        self.streaming = streaming
        self._client = client or requests.Session()
        self._validate_env_vars()

    def _validate_env_vars(self):
        if not self.sambaverse_model_name:
            self.sambaverse_model_name = os.getenv("SAMBAVERSE_MODEL_NAME")
        if not self.sambaverse_api_key:
            self.sambaverse_api_key = os.getenv("SAMBAVERSE_API_KEY")

        if not self.sambaverse_model_name:
            raise ValueError(
                "Sambaverse model name must be provided either as an argument or set in the environment variable 'SAMBAVERSE_MODEL_NAME'."
            )

        if not self.sambaverse_api_key:
            raise ValueError(
                "Sambaverse API key must be provided either as an argument or set in the environment variable 'SAMBAVERSE_API_KEY'."
            )

    def _get_full_url(self, endpoint: str) -> str:
        return f"{self.sambaverse_url}/{endpoint}"

    def _get_model_kwargs(self, stop: Optional[List[str]]) -> str:
        try:
            _model_kwargs = self.model_kwargs or {}
            _kwarg_stop_sequences = set(_model_kwargs.get("stop_sequences", []))
            _stop_sequences = set(stop or _kwarg_stop_sequences)

            if not _kwarg_stop_sequences:
                _model_kwargs["stop_sequences"] = ",".join(
                    f'"{x}"' for x in _stop_sequences
                )

            tuning_params_dict = {
                k: {"type": type(v).__name__, "value": str(v)}
                for k, v in _model_kwargs.items()
            }

            return json.dumps(tuning_params_dict)

        except Exception as e:
            raise ValueError(f"Error getting model kwargs: {e}")

    def _process_api_response(self, response: requests.Response) -> Dict:
        result: Dict[str, Any] = {}
        if response.status_code != 200:
            raise ValueError(
                f"Received unexpected status code {response.status_code}: {response.text}"
            )

        try:
            lines_result = response.text.strip().split("\n")
            text_result = lines_result[-1]
            if response.status_code == 200 and json.loads(text_result).get("error"):
                completion = ""
                for line in lines_result[:-1]:
                    completion += json.loads(line)["result"]["responses"][0][
                        "stream_token"
                    ]
                text_result = lines_result[-2]
                result = json.loads(text_result)
                result["result"]["responses"][0]["completion"] = completion
            else:
                result = json.loads(text_result)
        except Exception as e:
            result["detail"] = str(e)
        if "status_code" not in result:
            result["status_code"] = response.status_code
        return result

    def _process_api_stream_response(self, response: requests.Response) -> Any:
        try:
            for line in response.iter_lines():
                chunk = json.loads(line)
                if "status_code" not in chunk:
                    chunk["status_code"] = response.status_code
                if chunk["status_code"] == 200 and chunk.get("error"):
                    chunk["result"] = {"responses": [{"stream_token": ""}]}
                    return chunk
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Error processing streaming response: {e}")

    def _send_sambaverse_request(
        self, endpoint: str, data: Dict[str, Any], stream: bool = False
    ) -> requests.Response:
        url = self._get_full_url(endpoint)
        headers = {
            "key": self.sambaverse_api_key,
            "Content-Type": "application/json",
            "modelName": self.sambaverse_model_name,
        }
        try:
            return self._client.post(url, headers=headers, json=data, stream=stream)

        except Exception as e:
            raise ValueError(f"Error sending request to Sambaverse: {e}")

    def _prepare_request_data(self, prompt: str) -> Dict[str, Any]:
        try:
            model_params = self._get_model_kwargs(stop=None)
            return {"instance": prompt, "params": json.loads(model_params)}

        except Exception as e:
            raise ValueError(f"Error preparing request data: {e}")

    def _get_completion_from_response(self, response: Dict) -> str:
        try:
            return (
                response.get("result", {})
                .get("responses", [{}])[0]
                .get("completion", "")
            )
        except Exception as e:
            raise ValueError(f"Error processing response: {e}")

    @classmethod
    def class_name(cls) -> str:
        return "Samabaverse"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_name=self.sambaverse_model_name,
            model_kwargs=self.model_kwargs,
            description="Sambanova LLM",
            is_streaming=self.streaming,
        )

    @llm_completion_callback()
    def complete(self, prompt: str) -> CompletionResponse:
        """
        Complete the given prompt using the Sambaverse model.

        Args:
            prompt (str): The input prompt to complete.

        Returns:
            CompletionResponse: The completed text generated by the model.
        """
        data = self._prepare_request_data(prompt)
        response = self._send_sambaverse_request("api/predict", data)
        processed_response = self._process_api_response(response)
        completion_text = self._get_completion_from_response(processed_response)

        return CompletionResponse(text=completion_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str) -> CompletionResponseGen:
        """
        Stream the completion of the given prompt using the Sambaverse model.

        Args:
            prompt (str): The input prompt to complete.

        Yields:
            CompletionResponseGen: Streamed completion text generated by the model.
        """
        print("In stream_complete")
        data = self._prepare_request_data(prompt)
        response = self._send_sambaverse_request("api/predict", data, stream=True)

        for token in self._process_api_stream_response(response):
            processed_token = token["result"]["responses"][0]["stream_token"]
            yield CompletionResponse(text=processed_token)


class SambaStudio(CustomLLM):
    """
    SambaStudio LLM.

    Examples:
        `pip install llama-index-llms-sambanova`

        ```python
        from llama_index.llms.sambanova import SambaStudio

        llm = Sambaverse(...)

        response = llm.complete("What is the meaning of life?")

        print(response)
        ```
    """

    sambastudio_base_url: str = Field(
        default="",
        description="URL of the SambaStudio server",
    )
    sambastudio_base_uri: str = Field(
        default="",
        description="Base URI of the SambaStudio server",
    )
    sambastudio_project_id: str = Field(
        default="",
        description="Project ID of the SambaStudio server",
    )
    sambastudio_endpoint_id: str = Field(
        default="",
        description="Endpoint ID of the SambaStudio server",
    )
    sambastudio_api_key: str = Field(
        default="",
        description="API key for the SambaStudio server",
    )

    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the model",
    )
    streaming: bool = Field(
        default=False,
        description="Boolean to state whether to stream response or not",
    )

    _client: requests.Session = PrivateAttr()

    def __init__(
        self,
        sambastudio_base_url: str = "",
        sambastudio_base_uri: str = "",
        sambastudio_project_id: str = "",
        sambastudio_endpoint_id: str = "",
        model_kwargs: Dict[str, Any] = {},
        streaming: bool = False,
        client: Optional[requests.Session] = None,
    ) -> None:
        super().__init__()

        self.sambastudio_base_url = sambastudio_base_url
        self.sambastudio_base_uri = sambastudio_base_uri
        self.sambastudio_project_id = sambastudio_project_id
        self.sambastudio_endpoint_id = sambastudio_endpoint_id
        self.model_kwargs = model_kwargs
        self.streaming = streaming
        self._client = client or requests.Session()
        self._validate_env_vars()

    def _validate_env_vars(self):
        if not self.sambaverse_api_key:
            self.sambaverse_api_key = os.getenv("SAMBAVERSE_API_KEY")
        if not self.sambastudio_base_url:
            self.sambastudio_base_url = os.getenv("SAMBASTUDIO_BASE_URL")
        if not self.sambastudio_base_uri:
            self.sambastudio_base_uri = os.getenv("SAMBASTUDIO_BASE_URI")
        if not self.sambastudio_project_id:
            self.sambastudio_project_id = os.getenv("SAMBASTUDIO_PROJECT_ID")
        if not self.sambastudio_endpoint_id:
            self.sambastudio_endpoint_id = os.getenv("SAMBASTUDIO_ENDPOINT_ID")

        if not self.sambaverse_api_key:
            raise ValueError(
                "Sambaverse API key must be provided either as an argument or set in the environment variable 'SAMBAVERSE_API_KEY'."
            )

        if not self.sambastudio_base_url:
            raise ValueError(
                "Sambastudio base URL must be provided either as an argument or set in the environment variable 'SAMBASTUDIO_BASE_URL'."
            )

        if not self.sambastudio_base_uri:
            raise ValueError(
                "Sambastudio base URI must be provided either as an argument or set in the environment variable 'SAMBASTUDIO_BASE_URI'."
            )

        if not self.sambastudio_project_id:
            raise ValueError(
                "Sambastudio project ID must be provided either as an argument or set in the environment variable 'SAMBASTUDIO_PROJECT_ID'."
            )

        if not self.sambastudio_endpoint_id:
            raise ValueError(
                "Sambastudio endpoint ID must be provided either as an argument or set in the environment variable 'SAMBASTUDIO_ENDPOINT_ID'."
            )

    def _get_full_url(self, path: str) -> str:
        return f"{self.sambastudio_base_url}/{self.sambastudio_base_uri}/{path}"

    def _get_model_kwargs(self, stop: Optional[List[str]]) -> str:
        try:
            _model_kwargs = self.model_kwargs or {}
            _kwarg_stop_sequences = set(_model_kwargs.get("stop_sequences", []))
            _stop_sequences = set(stop or _kwarg_stop_sequences)

            if not _kwarg_stop_sequences:
                _model_kwargs["stop_sequences"] = ",".join(
                    f'"{x}"' for x in _stop_sequences
                )

            tuning_params_dict = {
                k: {"type": type(v).__name__, "value": str(v)}
                for k, v in _model_kwargs.items()
            }

            return json.dumps(tuning_params_dict)

        except Exception as e:
            raise ValueError(f"Error getting model kwargs: {e}")

    def _process_api_response(self, response: requests.Response) -> Dict:
        result: Dict[str, Any] = {}
        try:
            result = response.json()
        except Exception as e:
            result["detail"] = str(e)
        if "status_code" not in result:
            result["status_code"] = response.status_code
        return result

    def _process_api_stream_response(self, response: requests.Response) -> Any:
        """Process the streaming response."""
        if "nlp" in self.sambastudio_base_uri:
            try:
                import sseclient
            except ImportError:
                raise ImportError(
                    "could not import sseclient library"
                    "Please install it with `pip install sseclient-py`."
                )
            client = sseclient.SSEClient(response)
            close_conn = False
            for event in client.events():
                if event.event == "error_event":
                    close_conn = True
                chunk = {
                    "event": event.event,
                    "data": event.data,
                    "status_code": response.status_code,
                }
                yield chunk
            if close_conn:
                client.close()
        elif "generic" in self.sambastudio_base_uri:
            try:
                for line in response.iter_lines():
                    chunk = json.loads(line)
                    if "status_code" not in chunk:
                        chunk["status_code"] = response.status_code
                    if chunk["status_code"] == 200 and chunk.get("error"):
                        chunk["result"] = {"responses": [{"stream_token": ""}]}
                    yield chunk
            except Exception as e:
                raise RuntimeError(f"Error processing streaming response: {e}")
        else:
            raise ValueError(
                f"handling of endpoint uri: {self.api_base_uri} not implemented"
            )

    def _send_sambaverse_request(
        self, data: Dict[str, Any], stream: bool = False
    ) -> requests.Response:
        try:
            if stream:
                url = self._get_full_url(
                    f"stream/{self.sambastudio_project_id}/{self.sambastudio_endpoint_id}"
                )
                headers = {
                    "key": self.sambaverse_api_key,
                    "Content-Type": "application/json",
                }
                return self._client.post(url, headers=headers, json=data, stream=True)
            else:
                url = self._get_full_url(
                    f"{self.sambastudio_project_id}/{self.sambastudio_endpoint_id}"
                )
                headers = {
                    "key": self.sambaverse_api_key,
                    "Content-Type": "application/json",
                }

                return self._client.post(url, headers=headers, json=data, stream=stream)
        except Exception as e:
            raise ValueError(f"Error sending request to Sambaverse: {e}")

    def _prepare_request_data(self, prompt: str) -> Dict[str, Any]:
        try:
            data = {}
            if isinstance(prompt, str):
                input = [prompt]
            if "nlp" in self.api_base_uri:
                model_params = self._get_model_kwargs(stop=None)
                if model_params:
                    data = {"inputs": input, "params": json.loads(model_params)}
                else:
                    data = {"inputs": input}
            elif "generic" in self.api_base_uri:
                model_params = self._get_model_kwargs(stop=None)
                if model_params:
                    data = {"instance": input, "params": json.loads(model_params)}
                else:
                    data = {"instance": input}
            else:
                raise ValueError(
                    f"handling of endpoint uri: {self.api_base_uri} not implemented"
                )
            return data

        except Exception as e:
            raise ValueError(f"Error preparing request data: {e}")

    def _get_completion_from_response(self, response: Dict) -> str:
        try:
            if "nlp" in self.sambastudio_base_uri:
                return response["data"][0]["completion"]
            elif "generic" in self.sambastudio_base_uri:
                return response["predictions"][0]["completion"]
            else:
                raise ValueError(
                    f"handling of endpoint uri: {self.sambastudio_base_uri} not implemented"
                )
        except Exception as e:
            raise ValueError(f"Error processing response: {e}")

    def _get_stream_token_from_response(self, response: Dict) -> str:
        try:
            if "nlp" in self.sambastudio_base_uri:
                return response["data"]["stream_token"]
            elif "generic" in self.sambastudio_base_uri:
                return response["result"]["responses"][0]["stream_token"]
            else:
                raise ValueError(
                    f"handling of endpoint uri: {self.sambastudio_base_uri} not implemented"
                )
        except Exception as e:
            raise ValueError(f"Error processing response: {e}")

    @classmethod
    def class_name(cls) -> str:
        return "SambaStudio"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_kwargs=self.model_kwargs,
            description="Sambanova LLM",
            is_streaming=self.streaming,
        )

    @llm_completion_callback()
    def complete(self, prompt: str) -> CompletionResponse:
        """
        Complete the given prompt using the SambaStudio model.

        Args:
            prompt (str): The input prompt to complete.

        Returns:
            CompletionResponse: The completed text generated by the model.
        """
        data = self._prepare_request_data(prompt)
        response = self._send_sambaverse_request(data)
        processed_response = self._process_api_response(response)
        completion_text = self._get_completion_from_response(processed_response)

        return CompletionResponse(text=completion_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str) -> CompletionResponseGen:
        """
        Stream the completion of the given prompt using the SambaStudio model.

        Args:
            prompt (str): The input prompt to complete.

        Yields:
            CompletionResponseGen: Streamed completion text generated by the model.
        """
        print("In stream_complete")
        data = self._prepare_request_data(prompt)
        response = self._send_sambaverse_request(data, stream=True)

        for token in self._process_api_stream_response(response):
            processed_token = self._get_stream_token_from_response(token)
            yield CompletionResponse(text=processed_token)

    @llm_completion_callback()
    def nlp_prediction(self, prompt: str) -> CompletionResponse:
        """
        Perform NLP prediction for the given prompt using the SambaStudio model.

        Args:
            prompt (str): The input prompt to predict.

        Returns:
            CompletionResponse: The prediction result generated by the model.
        """
        return self.complete(prompt)

    @llm_completion_callback()
    def nlp_prediction_stream(self, prompt: str) -> CompletionResponseGen:
        """
        Stream NLP prediction for the given prompt using the SambaStudio model.

        Args:
            prompt (str): The input prompt to predict.

        Yields:
            CompletionResponseGen: Streamed prediction result generated by the model.
        """
        return self.stream_complete(prompt)
