"""AWS Kendra Retriever."""

from typing import List, Optional, Dict, Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.utilities.aws_utils import get_aws_service_client
import aioboto3


class AmazonKendraRetriever(BaseRetriever):
    """
    AWS Kendra retriever for LlamaIndex.

    See https://aws.amazon.com/kendra/ for more info.

    Args:
        index_id: Kendra Index ID.
        query_config: Configuration for querying Kendra.
        profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.
        region_name: The aws region e.g., `us-west-2`.
            Fallback to AWS_DEFAULT_REGION env variable or region specified in
            ~/.aws/config.
        aws_access_key_id: The aws access key id.
        aws_secret_access_key: The aws secret access key.
        aws_session_token: AWS temporary session token.

    Example:
        .. code-block:: python

            from llama_index.retrievers.kendra import AmazonKendraRetriever

            retriever = AmazonKendraRetriever(
                index_id="<kendra-index-id>",
                query_config={
                    "PageSize": 4,
                    "AttributeFilter": {
                        "EqualsTo": {
                            "Key": "tag",
                            "Value": {"StringValue": "space"}
                        }
                    }
                },
            )

    """

    # Mapping of Kendra confidence levels to float scores
    CONFIDENCE_SCORES = {
        "VERY_HIGH": 1.0,
        "HIGH": 0.8,
        "MEDIUM": 0.6,
        "LOW": 0.4,
        "NOT_AVAILABLE": 0.0,
    }

    def __init__(
        self,
        index_id: str,
        query_config: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._client = get_aws_service_client(
            service_name="kendra",
            profile_name=profile_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        # Create async session with the same credentials
        self._async_session = aioboto3.Session(
            profile_name=profile_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        self.index_id = index_id
        self.query_config = query_config or {}
        super().__init__(callback_manager)

    def _parse_response(self, response: Dict[str, Any]) -> List[NodeWithScore]:
        """Parse Kendra response into NodeWithScore objects."""
        node_with_score = []
        result_items = response.get("ResultItems", [])

        for result in result_items:
            text = ""
            metadata = {}

            # Extract text based on result type
            if result.get("Type") == "ANSWER":
                text = (
                    result.get("AdditionalAttributes", [{}])[0]
                    .get("Value", {})
                    .get("TextWithHighlightsValue", {})
                    .get("Text", "")
                )
            elif result.get("Type") == "DOCUMENT":
                text = result.get("DocumentExcerpt", {}).get("Text", "")

            # Extract metadata
            if "DocumentId" in result:
                metadata["document_id"] = result["DocumentId"]
            if "DocumentTitle" in result:
                metadata["title"] = result.get("DocumentTitle", {}).get("Text", "")
            if "DocumentURI" in result:
                metadata["source"] = result["DocumentURI"]

            # Only create nodes for results with actual content
            if text:
                # Convert Kendra's confidence score to float
                confidence = result.get("ScoreAttributes", {}).get(
                    "ScoreConfidence", "NOT_AVAILABLE"
                )
                score = self.CONFIDENCE_SCORES.get(confidence, 0.0)

                node_with_score.append(
                    NodeWithScore(
                        node=TextNode(
                            text=text,
                            metadata=metadata,
                        ),
                        score=score,
                    )
                )

        return node_with_score

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronous retrieve method."""
        query = query_bundle.query_str

        query_params = {
            "IndexId": self.index_id,
            "QueryText": query.strip(),
            **self.query_config,
        }

        response = self._client.query(**query_params)
        return self._parse_response(response)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronous retrieve method."""
        query = query_bundle.query_str

        query_params = {
            "IndexId": self.index_id,
            "QueryText": query.strip(),
            **self.query_config,
        }

        async with self._async_session.client("kendra") as client:
            response = await client.query(**query_params)
            return self._parse_response(response)
