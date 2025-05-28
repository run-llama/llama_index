"""Bedrock Retriever."""

from typing import List, Optional, Dict, Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.utilities.aws_utils import get_aws_service_client


class AmazonKnowledgeBasesRetriever(BaseRetriever):
    """
    `Amazon Bedrock Knowledge Bases` retrieval.

    See https://aws.amazon.com/bedrock/knowledge-bases for more info.

    Args:
        knowledge_base_id: Knowledge Base ID.
        retrieval_config: Configuration for retrieval.
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

            from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id="<knowledge-base-id>",
                retrieval_config={
                    "vectorSearchConfiguration": {
                        "numberOfResults": 4,
                        "overrideSearchType": "SEMANTIC",
                        "filter": {
                            "equals": {
                                "key": "tag",
                                "value": "space"
                            }
                        }
                    }
                },
            )

    """

    def __init__(
        self,
        knowledge_base_id: str,
        retrieval_config: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._client = get_aws_service_client(
            service_name="bedrock-agent-runtime",
            profile_name=profile_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        self.knowledge_base_id = knowledge_base_id
        self.retrieval_config = retrieval_config
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        response = self._client.retrieve(
            retrievalQuery={"text": query.strip()},
            knowledgeBaseId=self.knowledge_base_id,
            retrievalConfiguration=self.retrieval_config,
        )
        results = response["retrievalResults"]
        node_with_score = []
        for result in results:
            metadata = {}
            if "location" in result:
                metadata["location"] = result["location"]
            if "metadata" in result:
                metadata["sourceMetadata"] = result["metadata"]
            node_with_score.append(
                NodeWithScore(
                    node=TextNode(
                        text=result["content"]["text"],
                        metadata=metadata,
                    ),
                    score=result["score"] if "score" in result else 0,
                )
            )
        return node_with_score
