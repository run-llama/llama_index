"""Bedrock Retriever."""

from typing import List, Optional, Dict, Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.utilities.aws_utils import get_aws_service_client
import aioboto3  # NEW IMPORT


class AmazonKnowledgeBasesRetriever(BaseRetriever):
    """
    `Amazon Bedrock Knowledge Bases` retrieval.

    See https://aws.amazon.com/bedrock/knowledge-bases for more info.

    Args:
        knowledge_base_id: Knowledge Base ID.
        retrieval_config: Configuration for retrieval.
        knowledge_base_type: Type of knowledge base - "VECTOR" or "MANAGED".
            When set to "MANAGED" and no retrieval_config is provided, automatically
            uses managedSearchConfiguration. Default None (backward compatible).
        managed_search_config: Convenience parameter for managed knowledge bases.
            If provided, wraps it as {"managedSearchConfiguration": managed_search_config}
            and uses that as retrieval_config.
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

            # Vector knowledge base (default)
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

            # Managed knowledge base
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id="<knowledge-base-id>",
                knowledge_base_type="MANAGED",
                managed_search_config={
                    "numberOfResults": 5,
                },
            )

    """

    def __init__(
        self,
        knowledge_base_id: str,
        retrieval_config: Optional[Dict[str, Any]] = None,
        knowledge_base_type: Optional[str] = None,
        managed_search_config: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        # Keep existing sync client for backward compatibility
        self._client = get_aws_service_client(
            service_name="bedrock-agent-runtime",
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

        self.knowledge_base_id = knowledge_base_id
        self.retrieval_config = self._resolve_retrieval_config(
            retrieval_config, knowledge_base_type, managed_search_config
        )
        super().__init__(callback_manager)

    @staticmethod
    def _resolve_retrieval_config(
        retrieval_config: Optional[Dict[str, Any]],
        knowledge_base_type: Optional[str],
        managed_search_config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve the retrieval configuration based on provided parameters.

        Priority order:
        1. Explicit retrieval_config (highest priority, used as-is)
        2. managed_search_config convenience parameter
        3. Auto-config based on knowledge_base_type
        4. Default: managedSearchConfiguration (managed KB)
        """
        if retrieval_config is not None:
            return retrieval_config

        if managed_search_config is not None:
            return {"managedSearchConfiguration": managed_search_config}

        if knowledge_base_type == "MANAGED":
            return {"managedSearchConfiguration": {}}

        if knowledge_base_type == "VECTOR":
            return {"vectorSearchConfiguration": {}}

        # Default to managed when no type specified
        return {"managedSearchConfiguration": {}}

    def _parse_response(self, response: Dict[str, Any]) -> List[NodeWithScore]:
        """Parse Knowledge Base response into NodeWithScore objects."""
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

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronous retrieve method."""
        query = query_bundle.query_str

        response = self._client.retrieve(
            retrievalQuery={"text": query.strip()},
            knowledgeBaseId=self.knowledge_base_id,
            retrievalConfiguration=self.retrieval_config,
        )

        return self._parse_response(response)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronous retrieve method."""
        query = query_bundle.query_str

        async with self._async_session.client("bedrock-agent-runtime") as client:
            response = await client.retrieve(
                retrievalQuery={"text": query.strip()},
                knowledgeBaseId=self.knowledge_base_id,
                retrievalConfiguration=self.retrieval_config,
            )

            return self._parse_response(response)


def agentic_retrieve(
    knowledge_base_id: str,
    query: str,
    *,
    generate_response: bool = False,
    number_of_results: int = 5,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute AgenticRetrieveStream for intelligent multi-step retrieval.

    Standalone helper for complex queries needing query decomposition and
    managed reranking. Only works with MANAGED knowledge bases.
    Requires boto3 >= 1.43.0.

    Args:
        knowledge_base_id: The managed knowledge base ID.
        query: The user query.
        generate_response: Generate a cited answer (default False, returns chunks).
        number_of_results: Max results per retrieval iteration.
        region_name: AWS region.
        profile_name: AWS credentials profile.

    Returns:
        Dict with 'results' and optionally 'generatedResponse'.

    Example:
        .. code-block:: python

            from llama_index.retrievers.bedrock import agentic_retrieve

            result = agentic_retrieve(
                knowledge_base_id="ABCDEFGHIJ",
                query="What are the differences between S3 storage classes?",
            )
            for r in result["results"]:
                print(r["content"]["text"])

    """
    import boto3

    session_kwargs: Dict[str, Any] = {}
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    session = boto3.Session(**session_kwargs)

    from botocore.config import Config as BotocoreConfig

    client_kwargs: Dict[str, Any] = {
        "service_name": "bedrock-agent-runtime",
        "config": BotocoreConfig(user_agent_extra="llama-index/bedrock-kb"),
    }
    if region_name:
        client_kwargs["region_name"] = region_name
    client = session.client(**client_kwargs)

    response = client.agentic_retrieve_stream(
        messages=[{"content": {"text": query}, "role": "user"}],
        retrievers=[
            {
                "configuration": {
                    "knowledgeBase": {
                        "knowledgeBaseId": knowledge_base_id,
                        "retrievalOverrides": {"maxNumberOfResults": number_of_results},
                    }
                }
            }
        ],
        agenticRetrieveConfiguration={
            "foundationModelType": "MANAGED",
            "rerankingModelType": "MANAGED",
        },
        generateResponse=generate_response,
    )

    results = []
    generated_answer = ""
    citations = []

    for event in response.get("stream", []):
        if "result" in event:
            result_event = event["result"]
            results = result_event.get("results", [])
            if "generatedResponse" in result_event:
                gen_resp = result_event["generatedResponse"]
                generated_answer = gen_resp.get("answer", "")
                citations = gen_resp.get("citations", [])
        elif "responseEvent" in event:
            generated_answer += event["responseEvent"].get("text", "")

    output: Dict[str, Any] = {"results": results}
    if generate_response and generated_answer:
        output["generatedResponse"] = {
            "answer": generated_answer,
            "citations": citations,
        }
    return output
