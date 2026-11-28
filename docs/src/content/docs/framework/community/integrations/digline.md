---
title: Regression Testing a Query Engine With digline
---

[digline](https://github.com/digline/digline) is a Python-native evaluation engine for LLM output: assertions run against an approved baseline committed in your repository, and `digline compare` exits non-zero when a run is worse than the answers you signed off on. When a suite samples, it records the interval its own samples spanned and reports a drop that stays inside that interval as unchanged, so sampling wobble is not reported as a regression.

The target is a function, so a LlamaIndex query engine is evaluated in process — no server, no HTTP endpoint and no vector database.

### Installation and Setup

```sh
pip install digline
```

Tested against `llama-index-core>=0.14.24,<0.15`.

### The Query Engine Under Test

The engine is the one your application already has. Nothing about it changes to be evaluated — `HashEmbedding` is the example's local stand-in, so the default path needs no API key:

```python
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


def build_engine(llm: LLM) -> BaseQueryEngine:
    index = VectorStoreIndex.from_documents(
        documents(), embed_model=HashEmbedding()
    )
    return RetrieverQueryEngine.from_args(
        VectorIndexRetriever(index=index, similarity_top_k=1),
        llm=llm,
        text_qa_template=PromptTemplate(QA_TEMPLATE),
    )


def answer(question: str, llm: LLM) -> str:
    return str(build_engine(llm).query(question))
```

The index is rebuilt and queried on every run, so a change to the corpus, the chunking, the top-k or the embedding moves the numbers.

### The Suite

A `target` function calls the engine, and the suite declares what has to hold. `CITES_A_PAGE`, `JUDGE`, `CASES` and `_llm` are defined in the example's own `suite.py`, linked below:

```python
from pathlib import Path
from time import perf_counter

from digline.core import Faithfulness, Length, NotContains, Regex
from digline.run import Case, Response, Suite

import app


def target(case: Case) -> Response:
    question = str(case.vars["question"])
    started = perf_counter()
    said = app.answer(question, _llm())
    return Response(
        output=said,
        input=question,
        latency_ms=(perf_counter() - started) * 1000,
    )


suite = Suite(
    tenant="kestrel",
    environment="staging",
    name="hire",
    assertions=[
        Regex(pattern=CITES_A_PAGE, name="cites_a_page"),
        NotContains(needle="Question:", name="does_not_echo_the_prompt"),
        Length(minimum=60, name="says_something"),
        Faithfulness(judge=JUDGE, threshold=0.8, tolerance=0.1),
    ],
    cases=[
        Case(
            id=case["id"],
            vars={"question": case["question"]},
            context=[app.page(case["source"])],
        )
        for case in CASES
    ],
    artifacts=[Path("prompts/qa_template.txt")],
)
```

The prompt template is declared as an artifact, so every run records it and the report shows what changed in the prompt above the scores it moved.

### Grading Retrieval, Not Just Generation

Each case declares **the page that ought to answer it**, and that page — not what the retriever returned — is the case's context; grading against what was retrieved would be grading the run against itself. In the example, changing one line of the tokenizer, `SHORTEST` from 5 to 4, makes one question reach the hire page instead of the faults page. The answer is still well formed, still two sentences and still cites a page that exists, so the three string checks stay green and one check moves:

```
1 check got worse compared with the reference. Every case could be judged.
  brake-failed-mid-ride · faithfulness · Went from passing to failing (1.000000 → 0.000000).
```

Retrieval drift lands inside the graded check rather than beside it.

### The Cycle

```sh
digline run --suite suite.py
digline report --suite suite.py --run latest --locale en --out report.html
digline promote --suite suite.py --run latest
```

`promote` writes the baseline you commit; in CI, `run` then `compare` is the gate, and `compare` exits 0 when nothing got worse, 1 when something did and 2 when a case could not be judged.

### Useful Links

- [Example: a LlamaIndex query engine under test](https://github.com/digline/digline/tree/main/examples/llamaindex) — the complete, runnable source of the code above
- [Walkthrough of the example](https://digline.dev/product/examples/llamaindex/)
- [digline documentation](https://digline.dev/product/guide/)
