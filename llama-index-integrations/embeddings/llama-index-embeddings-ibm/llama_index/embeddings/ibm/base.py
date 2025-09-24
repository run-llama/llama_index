from typing import Any, List, Optional, Union, Dict

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr, SecretStr

from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.ibm.utils import (
    resolve_watsonx_credentials,
)

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings

DEFAULT_EMBED_MODEL = "ibm/slate-125m-english-rtrvr"


class WatsonxEmbeddings(BaseEmbedding):
    """
    IBM watsonx.ai embeddings.

    Example:
        `pip install llama-index-embeddings-ibm`

        ```python

        from llama_index.embeddings.ibm import WatsonxEmbeddings
        watsonx_llm = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="*****",
            project_id="*****",
        )
        ```

    """

    model_id: str = Field(
        default=DEFAULT_EMBED_MODEL,
        description="Type of model to use.",
        allow_mutation=False,
    )

    truncate_input_tokens: Optional[int] = Field(
        default=None,
        description="Represents the maximum number of input tokens accepted.",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="ID of the Watson Studio project.",
        allow_mutation=False,
    )

    space_id: Optional[str] = Field(
        default=None,
        description="ID of the Watson Studio space.",
        allow_mutation=False,
    )

    url: Optional[SecretStr] = Field(
        default=None,
        description="Url to the IBM watsonx.ai for IBM Cloud or the IBM watsonx.ai software instance.",
        allow_mutation=False,
    )

    apikey: Optional[SecretStr] = Field(
        default=None,
        description="API key to the IBM watsonx.ai for IBM Cloud or the IBM watsonx.ai software instance.",
        allow_mutation=False,
    )

    token: Optional[SecretStr] = Field(
        default=None,
        description="Token to the IBM watsonx.ai software instance.",
        allow_mutation=False,
    )

    password: Optional[SecretStr] = Field(
        default=None,
        description="Password to the IBM watsonx.ai software instance.",
        allow_mutation=False,
    )

    username: Optional[SecretStr] = Field(
        default=None,
        description="Username to the IBM watsonx.ai software instance.",
        allow_mutation=False,
    )

    instance_id: Optional[SecretStr] = Field(
        default=None,
        description="Instance_id of the IBM watsonx.ai software instance.",
        allow_mutation=False,
        deprecated="The `instance_id` parameter is deprecated and will no longer be utilized for logging to the IBM watsonx.ai software instance.",
    )

    version: Optional[SecretStr] = Field(
        default=None,
        description="Version of the IBM watsonx.ai software instance.",
        allow_mutation=False,
    )

    verify: Union[str, bool, None] = Field(
        default=None,
        description="""User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made""",
        allow_mutation=False,
    )

    # Enabled by default since IBM watsonx SDK 1.1.2 but it can cause problems
    # in environments where long-running connections are not supported.
    persistent_connection: bool = Field(
        default=True, description="Use persistent connection"
    )

    _embed_model: Embeddings = PrivateAttr()

    def __init__(
        self,
        model_id: str,
        truncate_input_tokens: Optional[int] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        url: Optional[str] = None,
        apikey: Optional[str] = None,
        token: Optional[str] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        version: Optional[str] = None,
        verify: Union[str, bool, None] = None,
        api_client: Optional[APIClient] = None,
        persistent_connection: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        callback_manager = callback_manager or CallbackManager([])

        if isinstance(api_client, APIClient):
            project_id = api_client.default_project_id or project_id
            space_id = api_client.default_space_id or space_id
            creds = {}
        else:
            creds = resolve_watsonx_credentials(
                url=url,
                apikey=apikey,
                token=token,
                username=username,
                password=password,
            )

        url = creds.get("url").get_secret_value() if creds.get("url") else None
        apikey = creds.get("apikey").get_secret_value() if creds.get("apikey") else None
        token = creds.get("token").get_secret_value() if creds.get("token") else None
        password = (
            creds.get("password").get_secret_value() if creds.get("password") else None
        )
        username = (
            creds.get("username").get_secret_value() if creds.get("username") else None
        )

        super().__init__(
            model_id=model_id,
            truncate_input_tokens=truncate_input_tokens,
            project_id=project_id,
            space_id=space_id,
            url=url,
            apikey=apikey,
            token=token,
            password=password,
            username=username,
            version=version,
            verify=verify,
            persistent_connection=persistent_connection,
            callback_manager=callback_manager,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

        self._embed_model = Embeddings(
            model_id=model_id,
            params=self.params,
            credentials=(
                Credentials.from_dict(
                    {
                        key: value.get_secret_value() if value else None
                        for key, value in self._get_credential_kwargs().items()
                    },
                    _verify=self.verify,
                )
                if creds
                else None
            ),
            project_id=self.project_id,
            space_id=self.space_id,
            api_client=api_client,
            persistent_connection=self.persistent_connection,
        )

    class Config:
        validate_assignment = True

    @classmethod
    def class_name(cls) -> str:
        return "WatsonxEmbedding"

    def _get_credential_kwargs(self) -> Dict[str, SecretStr | None]:
        return {
            "url": self.url,
            "apikey": self.apikey,
            "token": self.token,
            "password": self.password,
            "username": self.username,
            "version": self.version,
        }

    @property
    def params(self) -> Dict[str, int] | None:
        return (
            {"truncate_input_tokens": self.truncate_input_tokens}
            if self.truncate_input_tokens
            else None
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed_model.embed_query(text=query, params=self.params)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_query_embedding(query=text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed_model.embed_documents(texts=texts, params=self.params)

    ### Async methods
    # Asynchronous evaluation is not yet supported for watsonx.ai embeddings
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)
