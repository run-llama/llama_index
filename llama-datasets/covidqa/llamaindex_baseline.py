import asyncio

from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex


async def main():
    # DOWNLOAD LLAMADATASET
    rag_dataset, documents = download_llama_dataset("CovidQaDataset", "./data")

    # BUILD BASIC RAG PIPELINE
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()

    # EVALUATE WITH PACK
    RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
    rag_evaluator = RagEvaluatorPack(query_engine=query_engine, rag_dataset=rag_dataset)

    ############################################################################
    # NOTE: If have a lower tier subscription for OpenAI API like Usage Tier 1 #
    # then you'll need to use different batch_size and sleep_time_in_seconds.  #
    # For Usage Tier 1, settings that seemed to work well were batch_size=5,   #
    # and sleep_time_in_seconds=15 (as of December 2023.)                      #
    ############################################################################
    benchmark_df = await rag_evaluator.arun(
        batch_size=40,  # batches the number of openai api calls to make
        sleep_time_in_seconds=1,  # number of seconds sleep before making an api call
    )
    print(benchmark_df)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main)
