from .app_retriever import retriever
from llama_index.networks.contributor.retriever.service import (
    ContributorRetrieverService,
    ContributorRetrieverServiceSettings,
)


settings = ContributorRetrieverServiceSettings()
service = ContributorRetrieverService(config=settings, retriever=retriever)
app = service.app

# # Can add custom endpoints and security to app
# @app.get("/api/users/me/")
# async def custom_endpoint_logic():
#     ...
