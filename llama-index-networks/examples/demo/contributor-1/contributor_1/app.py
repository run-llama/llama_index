from .app_query_engine import query_engine
from llama_index.networks.contributor.query_engine.service import (
    ContributorQueryEngineService,
    ContributorQueryEngineServiceSettings,
)


settings = ContributorQueryEngineServiceSettings()
service = ContributorQueryEngineService(config=settings, query_engine=query_engine)
app = service.app

# # Can add custom endpoints and security to app
# @app.get("/api/users/me/")
# async def custom_endpoint_logic():
#     ...
