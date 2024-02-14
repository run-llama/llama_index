# TruLens-Eval Llama-Pack

![TruLens](https://www.trulens.org/assets/images/Neural_Network_Explainability.png)

The best way to support TruLens is to give us a ⭐ on [GitHub](https://www.github.com/truera/trulens) and join our [slack community](https://communityinviter.com/apps/aiqualityforum/josh)!

TruLens provides three Llamma Packs for LLM app observability:

- The first is the **TruLensRAGTriadPack** (context relevance, groundedness, answer relevance). This triad holds the key to detecting hallucination.

- Second, is the **TruLensHarmlessPack** including moderation and safety evaluations like criminality, violence and more.

- Last is the **TruLensHelpfulPack**, including evaluations like conciseness and language match.

No matter which TruLens LlamaPack you choose, all three provide evaluation and tracking for your LlamaIndex app with [TruLens](https://github.com/truera/trulens), an open-source LLM observability library from [TruEra](https://www.truera.com/).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack TruLensRAGTriadPack --download-dir ./trulens_pack
```

You can then inspect the files at `./trulens_pack` and use them as a template for your own project.

## Code Usage

You can download each pack to a `./trulens_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
TruLensRAGTriadPack = download_llama_pack(
    "TruLensRAGTriadPack", "./trulens_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./trulens_pack`.

Then, you can set up the pack like so:

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

from tqdm.auto import tqdm
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://paulgraham.com/worked.html"]
)

splitter = SentenceSplitter()
nodes = splitter.get_nodes_from_documents(documents)

trulens_ragtriad_pack = TruLensRAGTriadPack(
    nodes=nodes, app_id="Query Engine v1: RAG Triad Evals"
)
```

Then run your queries and evaluate!

```python
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in tqdm(queries):
    print("Query")
    print("=====")
    print(query)
    print()
    response = trulens_ragtriad_pack.run(query)
    print("Response")
    print("========")
    print(response)
```

You can access the internals of the LlamaPack, including your TruLens session and your query engine, via the `get_modules` method.

```python
modules = trulens_ragtriad_pack.get_modules()
tru = modules["session"]
index = modules["index"]
query_engine = modules["query_engine"]
tru_query_engine = modules["tru_query_engine"]
```

```python
tru.get_leaderboard(app_ids=["Query Engine v1: RAG Triad Evals"])
```

## Resources

There is a more complete notebook demo [available in the llama-hub repo](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/trulens_eval_packs/trulens_eval_llama_packs.ipynb).

Check out the [TruLens documentation](https://www.trulens.org/trulens_eval/install/) for more information!
