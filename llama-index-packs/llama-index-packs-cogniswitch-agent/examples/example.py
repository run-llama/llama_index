# Required Environment Variables: OPENAI_API_KEY, CS_TOKEN, CS_API_KEY
import os
from llama_index.packs.cogniswitch_agent import CogniswitchAgentPack

# Set up Cogniswitch Credentials
cogniswitch_tool_args = {
    "cs_token": os.getenv("CS_TOKEN"),
    "apiKey": os.getenv("CS_API_KEY"),
}

# create the pack
cogniswitch_agent_pack = CogniswitchAgentPack(cogniswitch_tool_args)

# run the pack (uploading the URL to Cogniswitch)
response = cogniswitch_agent_pack.run(
    "Upload this URL- https://cogniswitch.ai/developer"
)
print(response)
