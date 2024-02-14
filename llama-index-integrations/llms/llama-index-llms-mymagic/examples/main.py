from llama_index.llms.mymagic import MyMagicAI
import logging

logger = logging.getLogger(__name__)


model = MyMagicAI(
    api_key="test______________________Vitai",
    storage_provider="gcs",
    bucket_name="vitali-mymagic",
    session="test-session",
    system_prompt="Answer the question succinclty",
)

resp = model.complete(
    question="What is the CEO of the company?", model="mistral7b", max_tokens=10
)

print(resp)
