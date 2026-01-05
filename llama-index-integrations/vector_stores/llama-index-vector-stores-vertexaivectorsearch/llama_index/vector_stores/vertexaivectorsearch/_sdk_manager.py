from typing import Union

from google.oauth2.service_account import Credentials  # type: ignore
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import telemetry
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)

from llama_index.vector_stores.vertexaivectorsearch.utils import (
    get_client_info,
    get_user_agent,
)


class VectorSearchSDKManager:
    """
    Class in charge of building all Google Cloud SDK Objects needed to build
    VectorStores from project_id, credentials or other specifications. Abstracts
    away the authentication layer.
    """

    def __init__(
        self,
        *,
        project_id: str,
        region: str,
        credentials: Union[Credentials, None] = None,
        credentials_path: Union[str, None] = None,
    ) -> None:
        """
        Constructor.
        If `credentials` is provided, those credentials are used. If not provided
        `credentials_path` is used to retrieve credentials from a file. If also not
        provided, falls back to default credentials.

        Args:
            project_id: Id of the project.
            region: Region of the project. E.j. 'us-central1'
            credentials: Google cloud Credentials object.
            credentials_path: Google Cloud Credentials json file path.

        """
        self._project_id = project_id
        self._region = region

        if credentials is not None:
            self._credentials = credentials
        elif credentials_path is not None:
            self._credentials = Credentials.from_service_account_file(credentials_path)
        else:
            self._credentials = None

        # v2 client is initialized lazily
        self._v2_client = None
        self._v2_available = None

        self.initialize_aiplatform()

    def initialize_aiplatform(self) -> None:
        """Initializes aiplatform."""
        aiplatform.init(
            project=self._project_id,
            location=self._region,
            credentials=self._credentials,
        )

    def get_gcs_client(self) -> storage.Client:
        """
        Retrieves a Google Cloud Storage client.

        Returns:
            Google Cloud Storage Agent.

        """
        return storage.Client(
            project=self._project_id,
            credentials=self._credentials,
            client_info=get_client_info(
                module="llama-index-vector-stores-vertexaivectorsearch"
            ),
        )

    def get_gcs_bucket(self, bucket_name: str) -> storage.Bucket:
        """
        Retrieves a Google Cloud Bucket by bucket name.

        Args:
            bucket_name: Name of the bucket to be retrieved.

        Returns:
            Google Cloud Bucket.

        """
        client = self.get_gcs_client()
        return client.get_bucket(bucket_name)

    def get_index(self, index_id: str) -> MatchingEngineIndex:
        """
        Retrieves a MatchingEngineIndex (VectorSearchIndex) by id.

        Args:
            index_id: Id of the index to be retrieved.

        Returns:
            MatchingEngineIndex instance.

        """
        _, user_agent = get_user_agent("llama-index-vector-stores-vertexaivectorsearch")
        with telemetry.tool_context_manager(user_agent):
            return MatchingEngineIndex(
                index_name=index_id,
                project=self._project_id,
                location=self._region,
                credentials=self._credentials,
            )

    def get_endpoint(self, endpoint_id: str) -> MatchingEngineIndexEndpoint:
        """
        Retrieves a MatchingEngineIndexEndpoint (VectorSearchIndexEndpoint) by id.

        Args:
            endpoint_id: Id of the endpoint to be retrieved.

        Returns:
            MatchingEngineIndexEndpoint instance.

        """
        _, user_agent = get_user_agent("llama-index-vector-stores-vertexaivectorsearch")
        with telemetry.tool_context_manager(user_agent):
            return MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_id,
                project=self._project_id,
                location=self._region,
                credentials=self._credentials,
            )

    def is_v2_available(self) -> bool:
        """
        Check if v2 SDK is installed.

        Returns:
            bool: True if google-cloud-vectorsearch is available

        """
        if self._v2_available is None:
            import importlib.util

            self._v2_available = (
                importlib.util.find_spec("google.cloud.vectorsearch_v1beta") is not None
            )
        return self._v2_available

    def get_v2_client(self):
        """
        Get v2 clients only when needed - lazy import.

        Returns:
            dict: Dictionary containing the three v2 clients:
                - vector_search_service_client: For collection and index operations
                - data_object_service_client: For CRUD operations on data objects
                - data_object_search_service_client: For search/query operations

        Raises:
            ImportError: If google-cloud-vectorsearch is not installed

        """
        if self._v2_client is None:
            if not self.is_v2_available():
                raise ImportError(
                    "v2 requires 'google-cloud-vectorsearch'. "
                    "Install with: pip install 'llama-index-vector-stores-vertexaivectorsearch[v2]'"
                )

            # Import only when needed, not at module level
            from google.cloud import vectorsearch_v1beta

            # Initialize all three clients needed for v2
            self._v2_client = {
                "vector_search_service_client": vectorsearch_v1beta.VectorSearchServiceClient(
                    credentials=self._credentials
                ),
                "data_object_service_client": vectorsearch_v1beta.DataObjectServiceClient(
                    credentials=self._credentials
                ),
                "data_object_search_service_client": vectorsearch_v1beta.DataObjectSearchServiceClient(
                    credentials=self._credentials
                ),
            }
        return self._v2_client
