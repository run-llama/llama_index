<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# LlamaIndex Vector_Stores Integration: Vespa

[Vespa.ai](https://vespa.ai/) is an open-source big data serving engine. It is designed for low-latency and high-throughput serving of data and models. Vespa.ai is used by many companies to serve search results, recommendations, and rankings for billions of documents and users, expecting response times in the milliseconds.

This integration allows you to use Vespa.ai as a vector store for LlamaIndex. Vespa has integrated support for [embedding inference](https://docs.vespa.ai/en/embedding.html), so you don't need to run a separate service for these tasks.

Huggingface 🤗 embedders are supported, as well as SPLADE and ColBERT.

## Abstraction level of this integration

To make it really simple to get started, we provide a template Vespa application that will be deployed upon initializing the vector store. This removes some of the complexity of setting up Vespa for the first time, but for serious use cases, we strongly recommend that you read the [Vespa documentation](docs.vespa.ai) and tailor the application to your needs.

## The template

The provided template Vespa application can be seen below:

```python
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Component,
    Parameter,
    FieldSet,
    GlobalPhaseRanking,
    Function,
)

hybrid_template = ApplicationPackage(
    name="hybridsearch",
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary"]),
                    Field(
                        name="metadata", type="string", indexing=["summary"]
                    ),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                        bolding=True,
                    ),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[384])",
                        indexing=[
                            "input text",
                            "embed",
                            "index",
                            "attribute",
                        ],
                        ann=HNSW(distance_metric="angular"),
                        is_document_field=False,
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["text", "metadata"])],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[
                        Function(name="bm25sum", expression="bm25(text)")
                    ],
                    first_phase="bm25sum",
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                ),
                RankProfile(
                    name="fusion",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                        rerank_count=1000,
                    ),
                ),
            ],
        )
    ],
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"
                    },
                ),
            ],
        )
    ],
)
```

Note that the fields `id`, `metadata`, `text`, and `embedding` are required for the integration to work.
The schema name must also be `doc`, and the rank profiles must be named `bm25`, `semantic`, and `fusion`.

Other than that you are free to modify as you see fit by switching out embedding models, adding more fields, or changing the ranking expressions.

For more details, check out this Pyvespa example notebook on [hybrid search](https://pyvespa.readthedocs.io/en/latest/getting-started-pyvespa.html).

## Going to production

If you are ready to graduate to a production setup, we highly recommend to check out the [Vespa Cloud](https://cloud.vespa.ai/) service, where we manage all infrastructure and operations for you. Free trials are available.

## Next steps

There are many awesome features in Vespa, that are not exposed directly in this integration, check out [Pyvespa examples](https://pyvespa.readthedocs.io/en/latest/examples/pyvespa-examples.html) for some inspiration on what you can do with Vespa.

Teasers:

- Binary + Matryoshka embeddings.
- ColBERT.
- ONNX models.
- XGBoost and lightGBM models for ranking.
- Multivector indexing.
- and much more.
