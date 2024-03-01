from .app_query_engine import query_engine
from llama_index.networks.contributor.service import (
    ContributorService,
    ContributorServiceSettings,
)


settings = ContributorServiceSettings()
service = ContributorService(config=settings, query_engine=query_engine)
app = service.app

# # Can add custom endpoints and security to app
# @app.get("/api/users/me/")
# async def custom_endpoint_logic():
#     ...
