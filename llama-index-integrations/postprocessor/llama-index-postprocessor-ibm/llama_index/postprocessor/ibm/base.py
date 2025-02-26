from typing import Any, Dict, Optional, Union, List

from ibm_watsonx_ai import Credentials, APIClient

from ibm_watsonx_ai.foundation_models import Rerank
from ibm_watsonx_ai.foundation_models.schema import (
    RerankParameters,
    RerankReturnOptions,
)

from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
)

# Import SecretStr directly from pydantic
# since there is not one in llama_index.core.bridge.pydantic
from pydantic import SecretStr

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)

from llama_index.core.callbacks import CallbackManager

from llama_index.postprocessor.ibm.utils import resolve_watsonx_credentials

dispatcher = get_dispatcher(__name__)


class WatsonxRerank(BaseNodePostprocessor):
    """
    IBM watsonx.ai Rerank.

    Example:
        `pip install llama-index-postprocessor-ibm`

        ```python

        from llama_index.postprocessor.ibm import WatsonxRerank
        watsonx_llm = WatsonxRerank(
            model_id="<RERANK MODEL>",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="*****",
            project_id="*****",
        )
        ```
    """

    model_id: str = Field(description="Type of model to use.")

    top_n: Optional[int] = Field(
        default=None,
        description="Number of top results to return.",
    )

    truncate_input_tokens: Optional[int] = Field(
        default=None,
        description="""Represents the maximum number of input tokens accepted.""",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="ID of the Watson Studio project.",
        frozen=True,
    )

    space_id: Optional[str] = Field(
        default=None, description="ID of the Watson Studio space.", frozen=True
    )

    url: Optional[SecretStr] = Field(
        default=None,
        description="Url to Watson Machine Learning or CPD instance",
        frozen=True,
    )

    apikey: Optional[SecretStr] = Field(
        default=None,
        description="Apikey to Watson Machine Learning or CPD instance",
        frozen=True,
    )

    token: Optional[SecretStr] = Field(
        default=None, description="Token to CPD instance", frozen=True
    )

    password: Optional[SecretStr] = Field(
        default=None, description="Password to CPD instance", frozen=True
    )

    username: Optional[SecretStr] = Field(
        default=None, description="Username to CPD instance", frozen=True
    )

    instance_id: Optional[SecretStr] = Field(
        default=None, description="Instance_id of CPD instance", frozen=True
    )

    version: Optional[SecretStr] = Field(
        default=None, description="Version of CPD instance", frozen=True
    )

    verify: Union[str, bool, None] = Field(
        default=None,
        description="""
        User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made
        """,
        frozen=True,
    )

    _client: Optional[APIClient] = PrivateAttr()
    _watsonx_rerank: Rerank = PrivateAttr()

    def __init__(
        self,
        model_id: Optional[str] = None,
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        url: Optional[str] = None,
        apikey: Optional[str] = None,
        token: Optional[str] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        instance_id: Optional[str] = None,
        version: Optional[str] = None,
        verify: Union[str, bool, None] = None,
        api_client: Optional[APIClient] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize watsonx.ai Rerank.
        """
        callback_manager = callback_manager or CallbackManager([])

        creds = (
            resolve_watsonx_credentials(
                url=url,
                apikey=apikey,
                token=token,
                username=username,
                password=password,
                instance_id=instance_id,
            )
            if not isinstance(api_client, APIClient)
            else {}
        )

        super().__init__(
            model_id=model_id,
            top_n=top_n,
            truncate_input_tokens=truncate_input_tokens,
            project_id=project_id,
            space_id=space_id,
            url=creds.get("url"),
            apikey=creds.get("apikey"),
            token=creds.get("token"),
            password=creds.get("password"),
            username=creds.get("username"),
            instance_id=creds.get("instance_id"),
            version=version,
            verify=verify,
            _client=api_client,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = api_client
        self._watsonx_rerank = Rerank(
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
            verify=verify,
            api_client=api_client,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get Class Name."""
        return "WatsonxRerank"

    def _get_credential_kwargs(self) -> Dict[str, SecretStr | None]:
        return {
            "url": self.url,
            "apikey": self.apikey,
            "token": self.token,
            "password": self.password,
            "username": self.username,
            "instance_id": self.instance_id,
            "version": self.version,
        }

    @property
    def params(self) -> RerankParameters:
        rerank_return_options: RerankReturnOptions = RerankReturnOptions(
            top_n=self.top_n,
            inputs=False,
            query=False,
        )
        return RerankParameters(
            truncate_input_tokens=self.truncate_input_tokens,
            return_options=rerank_return_options,
        )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model_id,
            )
        )

        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        texts = [
            node.node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
        ]
        results = self._watsonx_rerank.generate(
            query=query_bundle.query_str,
            inputs=texts,
            params=self.params,
        )

        new_nodes = []
        for result in results.get("results", []):
            new_node_with_score = NodeWithScore(
                node=nodes[result["index"]].node,
                score=result["score"],
            )
            new_nodes.append(new_node_with_score)

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
