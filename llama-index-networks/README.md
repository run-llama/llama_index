# LlamaIndex Networks

The `llama-index-networks` library extension allows for the creation of
networks knowledge that can be queried and used for LLM context augmentation.
In such networks, we have "data suppliers" and "data consumers" that participate
and benefit from the network, respectively.

## Data suppliers

With this extension, you can easily wrap your llama-index `QueryEngine` with a
`ContributorService` that exposes the query engine behind a REST (FastAPI)
service, which renders it ready to contribute to LlamaIndex Network!

```python
from llama_index.networks.contributor import (
    ContributorServiceSettings,
    ContributorService,
)

query_engine = ...  # build a query engine as typically done with llama-index

settings = ContributorServiceSettings()
service = ContributorService(config=settings, query_engine=query_engine)
app = service.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
```

## Data consumers

With `llama-index-networks`, you can build a `NetworkQueryEngine` that is able
to connect to set of (network) `ContributorService`'s.

```python
from llama_index.networks.contributor import ContributorClient
from llama_index.networks.query_engine import NetworkQueryEngine
from llama_index.llms.groq import Groq

# Use ContributorClient to connect to a ContributorService
client = ContributorClient.from_config_file(env_file=".env.contributor.client")
contributors = [client]

# NetworkQueryEngine
llm = Groq()
network_query_engine = NetworkQueryEngine.from_args(
    contributors=contributors, llm=llm
)

# Can query it like any other query engine
network_query_engine.query("Why is the sky blue?")
```

The `.env.contributor.client` file contains the parameters to connect to the
`ContributorService` (i.e., `api_url` and `api_token`.)

## Examples

For a full demo, checkout the `examples/demo` sub-folder.

### FAQ

#### 1. Can I add my own custom endpoints to a `ContributorService`?

Yes, the (Fastapi) app is accessible via `ContributorService.app` and can be
modified to your needs.

```python
settings = ContributorServiceSettings()
service = ContributorService(config=settings, query_engine=query_engine)
app = service.app


@app.get("<custom-path>")
def custom_endpoint_logic():
    ...
```

#### 2. How can I add authentication to a `ContributorService`?

Currently, the client supports authentication through request headers as well as
as via an `api_token` str. On the server side, you should be able to modify the
app to your needs:

```python
settings = ContributorServiceSettings()
service = ContributorService(config=settings, query_engine=query_engine)
app = service.app

# modify app here to include your own security mechanisms
```

As we continue to build out this library extension, we welcome any feedback or
suggestions on what can be incorporated on this regard and others in order to
make our tools more helpful to our users.

## 📖 Citation

Reference to cite if you use LlamaIndex Networks in a paper, please cite both
the main library as well as this extension.

```latex
@software{Liu_LlamaIndex_2022,
author = {Liu, Jerry},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{LlamaIndex}},
url = {https://github.com/jerryjliu/llama_index},
year = {2022}
}

@software{Fajardo_LlamaIndexNetworks_2024,
author = {Fajardo, Val Andrei and Liu, Jerry and Markewich, Logan and Suo, Simon and Zhang, Haotian and Desai, Sourabh},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{LlamaIndex Networks}},
url = {https://github.com/run-llama/llama_index/llama-index-networks},
year = {2024}
}
```
