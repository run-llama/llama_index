import http
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
import urllib3
import vectorize_client as v
from urllib3 import HTTPResponse

from llama_index.core.schema import NodeRelationship
from llama_index.retrievers.vectorize import VectorizeRetriever

logger = logging.getLogger(__name__)

VECTORIZE_TOKEN = os.getenv("VECTORIZE_TOKEN", "")
VECTORIZE_ORG = os.getenv("VECTORIZE_ORG", "")


@pytest.fixture(scope="session")
def environment() -> Literal["prod", "dev", "local", "staging"]:
    env = os.getenv("VECTORIZE_ENV", "prod")
    if env not in ["prod", "dev", "local", "staging"]:
        msg = "Invalid VECTORIZE_ENV environment variable."
        raise ValueError(msg)
    return env


@pytest.fixture(scope="session")
def api_client(environment: str) -> Iterator[v.ApiClient]:
    header_name = None
    header_value = None
    if environment == "prod":
        host = "https://api.vectorize.io/v1"
    elif environment == "dev":
        host = "https://api-dev.vectorize.io/v1"
    elif environment == "local":
        host = "http://localhost:3000/api"
        header_name = "x-lambda-api-key"
        header_value = VECTORIZE_TOKEN
    else:
        host = "https://api-staging.vectorize.io/v1"

    with v.ApiClient(
        v.Configuration(host=host, access_token=VECTORIZE_TOKEN, debug=True),
        header_name,
        header_value,
    ) as api:
        yield api


@pytest.fixture(scope="session")
def pipeline_id(api_client: v.ApiClient) -> Iterator[str]:
    pipelines = v.PipelinesApi(api_client)

    connectors_api = v.ConnectorsApi(api_client)
    response = connectors_api.create_source_connector(
        VECTORIZE_ORG,
        [
            v.CreateSourceConnector(
                name="from api", type=v.SourceConnectorType.FILE_UPLOAD
            )
        ],
    )
    source_connector_id = response.connectors[0].id
    logger.info("Created source connector %s", source_connector_id)

    uploads_api = v.UploadsApi(api_client)
    upload_response = uploads_api.start_file_upload_to_connector(
        VECTORIZE_ORG,
        source_connector_id,
        v.StartFileUploadToConnectorRequest(
            name="research.pdf",
            content_type="application/pdf",
            metadata=json.dumps({"created-from-api": True}),
        ),
    )

    http_pool = urllib3.PoolManager()
    this_dir = Path(__file__).parent
    file_path = this_dir / "research.pdf"

    with file_path.open("rb") as f:
        http_response = http_pool.request(
            "PUT",
            upload_response.upload_url,
            body=f,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(file_path.stat().st_size),
            },
        )
    if http_response.status != http.HTTPStatus.OK:
        msg = "Upload failed:"
        raise ValueError(msg)
    else:
        logger.info("Upload successful")

    ai_platforms = connectors_api.get_ai_platform_connectors(VECTORIZE_ORG)
    builtin_ai_platform = next(
        c.id for c in ai_platforms.ai_platform_connectors if c.type == "VECTORIZE"
    )
    logger.info("Using AI platform %s", builtin_ai_platform)

    vector_databases = connectors_api.get_destination_connectors(VECTORIZE_ORG)
    builtin_vector_db = next(
        c.id for c in vector_databases.destination_connectors if c.type == "VECTORIZE"
    )
    logger.info("Using destination connector %s", builtin_vector_db)

    pipeline_response = pipelines.create_pipeline(
        VECTORIZE_ORG,
        v.PipelineConfigurationSchema(
            source_connectors=[
                v.SourceConnectorSchema(
                    id=source_connector_id,
                    type=v.SourceConnectorType.FILE_UPLOAD,
                    config={},
                )
            ],
            destination_connector=v.DestinationConnectorSchema(
                id=builtin_vector_db,
                type=v.DestinationConnectorType.VECTORIZE,
                config={},
            ),
            ai_platform=v.AIPlatformSchema(
                id=builtin_ai_platform,
                type=v.AIPlatformType.VECTORIZE,
                config=v.AIPlatformConfigSchema(),
            ),
            pipeline_name="Test pipeline",
            schedule=v.ScheduleSchema(type=v.ScheduleSchemaType.MANUAL),
        ),
    )
    pipeline_id = pipeline_response.data.id
    logger.info("Created pipeline %s", pipeline_id)

    yield pipeline_id

    try:
        pipelines.delete_pipeline(VECTORIZE_ORG, pipeline_id)
    except Exception:
        logger.exception("Failed to delete pipeline %s", pipeline_id)


@pytest.mark.skipif(
    VECTORIZE_TOKEN == "" or VECTORIZE_ORG == "",
    reason="missing Vectorize credentials (VECTORIZE_TOKEN, VECTORIZE_ORG)",
)
def test_retrieve_integration(
    environment: Literal["prod", "dev", "local", "staging"],
    pipeline_id: str,
) -> None:
    retriever = VectorizeRetriever(
        environment=environment,
        api_token=VECTORIZE_TOKEN,
        organization=VECTORIZE_ORG,
        pipeline_id=pipeline_id,
        num_results=2,
    )
    start = time.time()
    while True:
        docs = retriever.retrieve("What are you?")
        if len(docs) == 2:
            break
        if time.time() - start > 180:
            msg = "Docs not retrieved in time"
            raise RuntimeError(msg)
        time.sleep(1)


def test_retrieve_unit() -> None:
    retriever = VectorizeRetriever(
        environment="prod",
        api_token="fake_token",  # noqa: S106
        organization="fake_org",
        pipeline_id="fake_pipeline_id",
    )
    retriever._pipelines.api_client.rest_client.pool_manager.urlopen = MagicMock(
        return_value=HTTPResponse(
            body=json.dumps(
                {
                    "documents": [
                        {
                            "relevancy": 0.42,
                            "id": "fake_id",
                            "text": "fake_text",
                            "chunk_id": "fake_chunk_id",
                            "total_chunks": "fake_total_chunks",
                            "origin": "fake_origin",
                            "origin_id": "fake_origin_id",
                            "similarity": 0.43,
                            "source": "fake_source",
                            "unique_source": "fake_unique_source",
                            "source_display_name": "fake_source_display_name",
                        },
                    ],
                    "question": "fake_question",
                    "average_relevancy": 0.44,
                    "ndcg": 0.45,
                }
            ).encode(),
            status=200,
        )
    )
    docs = retriever.retrieve("fake_question")
    assert len(docs) == 1
    assert docs[0].node.id_ == "fake_id"
    assert docs[0].node.text == "fake_text"
    assert (
        docs[0].node.relationships[NodeRelationship.SOURCE].node_id
        == "fake_unique_source"
    )
    assert docs[0].score == 0.43
