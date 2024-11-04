DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
UPSERT_PIPELINE_ENDPOINT = "/api/v1/indices/pipeline"
START_PIPELINE_ENDPOINT = "/api/v1/indices/pipeline/{pipeline_id}/managed_ingest"
CHECK_INGESTION_ENDPOINT = (
    "/api/v1/indices/pipeline/{pipeline_id}/managed_ingest/{ingestion_id}/status"
)
RETRIEVE_PIPELINE_ENDPOINT = "/api/v1/indices/pipeline/{pipeline_id}/retrieve"
PIPELINE_SIMPLE_ENDPOINT = "/api/v1/indices/pipeline_simple"
INSERT_DOC_ENDPOINT = "/api/v1/indices/pipeline/{pipeline_id}/documents"
DELETE_DOC_ENDPOINT = "/api/v1/indices/pipeline/{pipeline_id}/delete"
