import asyncio

from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.llms import OpenAI, Gemini
from llama_index.core import ServiceContext
import pandas as pd


async def main():
    # DOWNLOAD LLAMADATASET
    pairwise_evaluator_dataset, _ = download_llama_dataset(
        "MtBenchHumanJudgementDataset", "./mt_bench_data"
    )

    # DEFINE EVALUATORS
    gpt_4_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    gpt_3p5_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
    )

    gemini_pro_context = ServiceContext.from_defaults(
        llm=Gemini(model="models/gemini-pro", temperature=0)
    )

    evaluators = {
        "gpt-4": PairwiseComparisonEvaluator(service_context=gpt_4_context),
        "gpt-3.5": PairwiseComparisonEvaluator(service_context=gpt_3p5_context),
        "gemini-pro": PairwiseComparisonEvaluator(service_context=gemini_pro_context),
    }

    # EVALUATE WITH PACK
    ############################################################################
    # NOTE: If have a lower tier subscription for OpenAI API like Usage Tier 1 #
    # then you'll need to use different batch_size and sleep_time_in_seconds.  #
    # For Usage Tier 1, settings that seemed to work well were batch_size=5,   #
    # and sleep_time_in_seconds=15 (as of December 2023.)                      #
    ############################################################################
    EvaluatorBenchmarkerPack = download_llama_pack("EvaluatorBenchmarkerPack", "./pack")
    evaluator_benchmarker = EvaluatorBenchmarkerPack(
        evaluator=evaluators["gpt-3.5"],
        eval_dataset=pairwise_evaluator_dataset,
        show_progress=True,
    )
    gpt_3p5_benchmark_df = await evaluator_benchmarker.arun(
        batch_size=100, sleep_time_in_seconds=0
    )

    evaluator_benchmarker = EvaluatorBenchmarkerPack(
        evaluator=evaluators["gpt-4"],
        eval_dataset=pairwise_evaluator_dataset,
        show_progress=True,
    )
    gpt_4_benchmark_df = await evaluator_benchmarker.arun(
        batch_size=100, sleep_time_in_seconds=0
    )

    evaluator_benchmarker = EvaluatorBenchmarkerPack(
        evaluator=evaluators["gemini-pro"],
        eval_dataset=pairwise_evaluator_dataset,
        show_progress=True,
    )
    gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
        batch_size=5, sleep_time_in_seconds=0.5
    )

    benchmark_df = pd.concat(
        [
            gpt_3p5_benchmark_df,
            gpt_4_benchmark_df,
            gemini_pro_benchmark_df,
        ],
        axis=0,
    )
    print(benchmark_df)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main)
