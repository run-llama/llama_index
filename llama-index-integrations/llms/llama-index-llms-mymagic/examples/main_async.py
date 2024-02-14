import asyncio
from llama_index.llms.mymagic import MyMagicAI


async def main():
    model = MyMagicAI(
        api_key="test______________________Vitai",
        storage_provider="gcs",
        bucket_name="vitali-mymagic",
        session="test-session",
        system_prompt="Answer the question succinclty",
    )

    resp = await model.acomplete(
        question="What is the CEO of the company?", model="mistral7b", max_tokens=10
    )

    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
