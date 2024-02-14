import asyncio
from llama_index.llms.mymagic import MyMagicAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    model = MyMagicAI(
        api_key="test______________________Vitai"
    )

    # `await` must be used to call the async method `acomplete`
    resp = await model.acomplete(
        question="What is the capital of France?",
    )

    print(resp)

# Run the async function using asyncio.run() if this is your script's entry point
if __name__ == "__main__":
    asyncio.run(main())
