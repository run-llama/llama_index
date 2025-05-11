import asyncio
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.gestell import GestellToolSpec

async def main():
    # 1. Load your .env, make sure OPENAI_API_KEY, GESTELL_API_KEY and GESTELL_COLLECTION_ID are available
    load_dotenv()

    # 2. Instantiate the Gestell tool spec
    tool_spec = GestellToolSpec()

    # 3. Generate the list of tools from that instance (can optionally add other tools for the agent to use)
    tool_list = tool_spec.to_tool_list()

    # 4. Create the OpenAI agent with these tools
    agent = OpenAIAgent.from_tools(tool_list, verbose=True)

    # 5. Run your queries
    response = await agent.achat(
        "Give me a concise summary of the documents in this collection."
    )
    print(response)

asyncio.run(main())
