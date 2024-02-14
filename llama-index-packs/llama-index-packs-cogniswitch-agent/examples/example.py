# Required Environment Variables: OPENAI_API_KEY, CS_TOKEN, CS_API_KEY
import os
from llama_index.core.llama_pack import download_llama_pack

# Set up Cogniswitch Credentials
cogniswitch_tool_args = {
    "cs_token": os.getenv("CS_TOKEN"),
    "apiKey": os.getenv("CS_API_KEY"),
}

# download and install dependencies
CogniswitchAgentPack = download_llama_pack(
    "CogniswitchAgentPack", "./cogniswitch_agent_pack"
)

# create the pack
cogniswitch_agent_pack = CogniswitchAgentPack(cogniswitch_tool_args)

# run the pack (uploading the URL to Cogniswitch)
response = cogniswitch_agent_pack.run(
    "Upload this URL- https://cogniswitch.ai/developer"
)
print(response)
