# Mini Mt Bench Single Grading Dataset

## CLI Usage

You can download `llamadatasets` directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamadataset MiniMtBenchSingleGradingDataset --download-dir ./data
```

You can then inspect the files at `./data`. When you're ready to load the data into
python, you can use the below snippet of code:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledEvaluatorDataset

evaluator_dataset = LabelledEvaluatorDataset.from_json(
    "./data/pairwise_evaluation_dataset.json"
)
```

## Code Usage

You can download the dataset to a directory, say `./data` directly in Python
as well. From there, you can use the convenient `EvaluatorBenchmarkerPack` llamapack to
run your own LlamaIndex RAG pipeline with the `llamadataset`.

```python
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms import OpenAI
from llama_index.core import ServiceContext

# download benchmark dataset
evaluator_dataset, _ = download_llama_dataset(
    "MiniMtBenchSingleGradingDataset", "./data"
)

# define your evaluator
gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0, model="gpt-4"),
)

evaluator = CorrectnessEvaluator(service_context=gpt_4_context)

# evaluate using the EvaluatorBenchmarkerPack
EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)
evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluator,
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

############################################################################
# NOTE: If have a lower tier subscription for OpenAI API like Usage Tier 1 #
# then you'll need to use different batch_size and sleep_time_in_seconds.  #
# For Usage Tier 1, settings that seemed to work well were batch_size=5,   #
# and sleep_time_in_seconds=15 (as of December 2023.)                      #
############################################################################

benchmark_df = await evaluator_benchmarker.arun(
    batch_size=20,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)
```

## Original data citation

```text
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
