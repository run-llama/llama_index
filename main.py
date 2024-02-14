from llama_index.llms.mymagic import MyMagicAI
import logging
logger = logging.getLogger(__name__)


model = MyMagicAI(
    api_key="test______________________Vitai"
    )

resp = model.acomplete(
    question="What is the capital of France?",
    )

print(resp)
