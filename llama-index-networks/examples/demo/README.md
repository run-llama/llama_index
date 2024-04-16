# LlamaIndex Networks Demo

In this demo, we have 3 network contributor services, whose Python source
codes can be found in the folders listed below:

- contributor-1/
- contributor-2/
- contributor-3/

Once all of these services are up and running (see usage instructions
attached below), then we can connect to them using a `NetworkQueryEngine`.
The demo for doing that is found in the notebook: `network_query_engine.ipynb`.

## Building and Running the 3 Network Contributor Services

### Setup Environment Variables

Each of the three Contributor Services wrap `QueryEngine`'s that utilize
OpenAI as the LLM provider. As such, you'll need to supply an `OPENAI_API_KEY`.

To do so, we make use of .env files. Each contributor folder requires a filled
in `.env.contributor.service` file. You can use the `template.env.contributor.service`,
fill in your openai-api-key and then save it as `.env.contributor.service` (you can also save it simply as `.env` as the `ContributorServiceSettings`
class will look for `.env` file if it can't find `.env.contributor.service`).

### Installation

_Requires Poetry v. 1.4.1 to be installed._

```sh
cd contributor-1 && poetry install && cd -
cd contributor-2 && poetry install && cd -
cd contributor-3 && poetry install && cd -
```

### Running the contributor servers locally

_Requires Docker to be installed._

We've simplified running all three services with the help of
`docker-compose`. It should be noted that in a real-world scenario, these
contributor services are likely independently stood up.

```sh
docker-compose up --build
```

Any code changes will be reflected in the running server automatically without having to rebuild/restart the server.

### Viewing the SWAGGER docs (local)

Once the server is running locally, the standard API docs of any of
the three contributor services can be viewed in a browser.

Use any port number `{8001,8002,8003}`:

```sh
# visit in any browser
http://localhost:<PORT>/docs#/
```

## Building and Running the NetworkQueryEngine

We begin by creating a fresh environment:

```sh
pyenv virtualenv networks-demo
pyenv activate networks-demo
pip install jupyterlab ipykernel
pip install llama-index llama-index-networks --pre
```

Let's create the kernel for our notebook so that we can use our newly
created `networks-demo` virtual environment.

```sh
ipython kernel install --user --name=networks-demo
```

### Setting up the environment files

As in setting up the `ContributorService`'s we need to pass in the settings
for the `ContributorClient`'s (that communicate with their respective services).
Simply rename the template files in `client-env-files` directory by dropping
the term "template" in all of the .env files (e.g.,
`template.env.contributor_1.client` becomes `.env.contributor_1.client`).

### Running the Notebook

Note this notebook uses OpenAI LLMs. We export the `OPENAI_API_KEY`
at the same time that we spin up the notebook:

```sh
export OPENAI_API_KEY=<openai-api-key> && jupyter lab
```

From there you can open up the `network_query_engine.ipynb`.
