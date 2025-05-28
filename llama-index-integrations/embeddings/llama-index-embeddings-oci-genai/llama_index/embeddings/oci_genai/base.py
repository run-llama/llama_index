from enum import Enum
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

SUPPORTED_MODELS = {
    "cohere.embed-english-v3.0",
    "cohere.embed-english-light-v3.0",
    "cohere.embed-multilingual-v3.0",
    "cohere.embed-multilingual-light-v3.0",
    "cohere.embed-english-light-v2.0",
}


class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"


class OCIGenAIEmbeddings(BaseEmbedding):
    """
    OCI embedding models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service. If a specific config profile is used,
    you must pass the name of the profile (~/.oci/config) through auth_profile.
    If a specific config file location is used, you must pass
    the file location where profile name configs present
    through auth_file_location

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from llama_index.embeddings.oci_genai import OCIGenAIEmbeddings

            embeddings = OCIGenAIEmbeddings(
                model_name="MY_EMBEDDING_MODEL",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID"
            )

    """

    model_name: str = Field(
        description="ID or Name of the OCI Generative AI embedding model to use."
    )

    truncate: str = Field(
        description="Truncate embeddings that are too long from start or end, values START/ END/ NONE",
        default="END",
    )

    input_type: Optional[str] = Field(
        description="Model Input type. If not provided, search_document and search_query are used when needed.",
        default=None,
    )

    service_endpoint: Optional[str] = Field(
        description="service endpoint url.",
        default=None,
    )

    compartment_id: Optional[str] = Field(
        description="OCID of compartment.",
        default=None,
    )

    auth_type: Optional[str] = Field(
        description="Authentication type, can be: API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL. If not specified, API_KEY will be used",
        default="API_KEY",
    )

    auth_profile: Optional[str] = Field(
        description="The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used",
        default="DEFAULT",
    )

    auth_file_location: Optional[str] = Field(
        description="Path to the config file. If not specified, ~/.oci/config will be used",
        default="~/.oci/config",
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        truncate: str = "END",
        input_type: Optional[str] = None,
        service_endpoint: Optional[str] = None,
        compartment_id: Optional[str] = None,
        auth_type: Optional[str] = "API_KEY",
        auth_profile: Optional[str] = "DEFAULT",
        auth_file_location: Optional[str] = "~/.oci/config",
        client: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """
        Initializes the OCIGenAIEmbeddings class.

        Args:
            model_name (str): The name or ID of the model to be used for generating embeddings, e.g., "cohere.embed-english-light-v3.0".

            truncate (str): A string indicating the truncation strategy for long input text. Possible values
                            are 'START', 'END', or 'NONE'.

            input_type (Optional[str]): An optional string that specifies the type of input provided to the model.
                                        This is model-dependent and could be one of the following: "search_query",
                                        "search_document", "classification", or "clustering".

            service_endpoint (str): service endpoint url, e.g., "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

            compartment_id (str): OCID of the compartment.

            auth_type (Optional[str]): Authentication type, can be: API_KEY (default), SECURITY_TOKEN, INSTANCEAL, RESOURCE_PRINCIPAL. If not specified, API_KEY will be used

            auth_profile (Optional[str]): The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used

            auth_file_location (Optional[str]): Path to the config file, If not specified, ~/.oci/config will be used.

            client (Optional[Any]): An optional OCI client object. If not provided, the client will be created using the
                                    provided service endpoint and authentifcation method.

        """
        super().__init__(
            model_name=model_name,
            truncate=truncate,
            input_type=input_type,
            service_endpoint=service_endpoint,
            compartment_id=compartment_id,
            auth_type=auth_type,
            auth_profile=auth_profile,
            auth_file_location=auth_file_location,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
        )
        if client is not None:
            self._client = client
        else:
            try:
                import oci

                client_kwargs = {
                    "config": {},
                    "signer": None,
                    "service_endpoint": service_endpoint,
                    "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
                    "timeout": (
                        10,
                        240,
                    ),  # default timeout config for OCI Gen AI service
                }

                if auth_type == OCIAuthType(1).name:
                    client_kwargs["config"] = oci.config.from_file(
                        file_location=auth_file_location, profile_name=auth_profile
                    )
                    client_kwargs.pop("signer", None)
                elif auth_type == OCIAuthType(2).name:

                    def make_security_token_signer(oci_config):  # type: ignore[no-untyped-def]
                        pk = oci.signer.load_private_key_from_file(
                            oci_config.get("key_file"), None
                        )
                        with open(
                            oci_config.get("security_token_file"), encoding="utf-8"
                        ) as f:
                            st_string = f.read()
                        return oci.auth.signers.SecurityTokenSigner(st_string, pk)

                    client_kwargs["config"] = oci.config.from_file(
                        file_location=auth_file_location, profile_name=auth_profile
                    )
                    client_kwargs["signer"] = make_security_token_signer(
                        oci_config=client_kwargs["config"]
                    )
                elif auth_type == OCIAuthType(3).name:
                    client_kwargs["signer"] = (
                        oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                    )
                elif auth_type == OCIAuthType(4).name:
                    client_kwargs["signer"] = (
                        oci.auth.signers.get_resource_principals_signer()
                    )
                else:
                    raise ValueError(
                        f"Please provide valid value to auth_type, {auth_type} is not valid."
                    )

                self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    **client_kwargs
                )

            except ImportError as ex:
                raise ModuleNotFoundError(
                    "Could not import oci python package. "
                    "Please make sure you have the oci package installed."
                ) from ex
            except Exception as e:
                raise ValueError(
                    """Could not authenticate with OCI client.
                    If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used, please check the specified
                    auth_profile, auth_file_location and auth_type are valid.""",
                    e,
                ) from e

    @classmethod
    def class_name(self) -> str:
        return "OCIGenAIEmbeddings"

    @staticmethod
    def list_supported_models() -> List[str]:
        return list(SUPPORTED_MODELS)

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        try:
            from oci.generative_ai_inference import models

        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        if self.model_name.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_name)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_name)

        request = models.EmbedTextDetails(
            serving_mode=serving_mode,
            compartment_id=self.compartment_id,
            input_type=self.input_type or input_type,
            truncate=self.truncate,
            inputs=texts,
        )

        response = self._client.embed_text(request)

        return response.data.embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], input_type="SEARCH_QUERY")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], input_type="SEARCH_DOCUMENT")[0]

    def _get_text_embeddings(self, text: str) -> List[List[float]]:
        return self._embed(text, input_type="SEARCH_DOCUMENT")

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
