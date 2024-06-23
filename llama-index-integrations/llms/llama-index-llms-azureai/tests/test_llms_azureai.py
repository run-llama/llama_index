from llama_index.llms.azureai.inference import AzureAIModelInference
from llama_index.core.llms import ChatMessage

llm = AzureAIModelInference(
    endpoint="https://Phi-3-mini-4k-instruct-iscuv-serverless.eastus2.inference.ai.azure.com",
    credential="qyMzcewyHfFPfL2FAi8Bh5xMefITUv4V",
)

llm.chat([ChatMessage(role="user", content="Hello")])
