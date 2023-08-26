from llama_index.storage import StorageContext
from llama_index import load_index_from_storage
from llama_index.llms import Portkey
from llama_index import ServiceContext
#  > provider: Optional[ProviderTypes]
# > model: str
# > temperature: float
# > max_tokens: Optional[int]
# > max_retries: int
# > trace_id: Optional[str]
# > cache_status: Optional[RubeusCacheType]
# > cache: Optional[bool]
# > metadata: Dict[str, Any]
# > weight: Optional[float]
llm = Portkey(mode="fallback", api_key="x2trk").add_llms(llm_params=[{
    "provider": "openai",
    "model_api_key": ""
}])

service_context = ServiceContext.from_defaults(llm=llm)

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context, service_context=service_context)

print('Starts the query engine here..')
query_engine = index.as_chat_engine(service_context=service_context)

print('Starts the query here..')
response = query_engine.chat("What did the author do after his time at Y Combinator?")
print(response)
