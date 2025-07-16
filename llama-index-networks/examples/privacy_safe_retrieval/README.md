# Privacy-Safe Network Retrieval Demo

In this demo, we showcase a privacy-safe network of retrievers, where the
the data that is exchanged is differentially-private, synthetic data. Data
collaboration can be immensely beneficial for downstream tasks such as better
insights or modelling. However, data that is sensitive may not be permitted to
share in such networks. Thus, an important avenue of research is towards creating
privacy-preserving techniques that would permit the safe exchange of such datasets,
without doing any privacy harm to the data subjects, while still maintaining the
utility of the dataset.

## The Data

The original data is the [Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) dataset.
The synthetic dataset was created by the `DiffPrivateSimpleDatasetPack` llama-pack.
For details on the dataset curation, see the demo in the Github repository for
that llama-pack (found [here](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-diff-private-simple-dataset/examples/symptom_2_disease)).

We created two versions of synthetic dataset, differing by the levels of privacy.
In particular, we have a synthetic dataset with epsilon = 1.3, and another one
where epsilon = 15.9. The epsilon value can be interpreted as privacy loss of a
data subject, and so a higher epsilon means less privacy.

The synthetic datasets have been stored in our dropbox and are downloaded within
the execution of `data_prep/create_contributor_synthetic_data.py`.

## The Network

In this demo, we have 2 contributor retriever services, whose Python source
codes can be found in the folders listed below:

- contributor-1/
- contributor-2/

To reiterate, the contributor retrievers are built on top of their own sets
of privacy-safe, synthetic examples of the Symptom2Disease dataset. Thus, the
data they share across the network is ``de-anonymized''.

Once all of these services are up and running (see usage instructions
attached below), then we can connect to them using a `NetworkRetrieverEngine`.
The demo for doing that is found in the notebook: `network_retriever.ipynb`.

## Building and Running the 2 Network Contributor Services

### Virtual Environment

We begin by creating a fresh environment:

```sh
pyenv virtualenv networks-retriever-demo
pyenv activate networks-retriever-demo
pip install jupyterlab ipykernel
pip install llama-index llama-index-networks
```

### Download The Data And Create Datasets For Each Contributor

With the `networks-retriever-demo` virtualenv activated:

```sh
cd privacy_safe_retrieval
python data_prep/create_contributor_synthetic_data.py
```

The output of this script will be the following four datasets:

- `data_prep/synthetic_dataset.json`
- `./symptom_2_disease_test.json`
- `contributor-1/data/contributor1_synthetic_dataset.json`
- `contributor-2/data/contributor2_synthetic_dataset.json`

### Setup Environment Variables

Each of the two Contributor Services wrap `Retriever`'s that utilize
OpenAI embeddings. As such, you'll need to supply an `OPENAI_API_KEY`.

To do so, we make use of .env files. Each contributor folder requires a filled
in `.env.contributor.service` file. You can use the `template.env.contributor.service`,
fill in your openai-api-key and then save it as `.env.contributor.service`
(you can also save it simply as `.env` as the `ContributorRetrieverServiceSettings`
class will look for `.env` file if it can't find `.env.contributor.service`).

Additionally, we need to define the `SIMILARITY_TOP_K` environment variable
for each of the retrievers. To do this, you can use `template.env.retriever` file
and fill in your desired top-k value and then save it as `.env.retriever`. You
must do this for both contributors.

### Install The Contributor Project Dependencies

_Requires Poetry v. 1.4.1 to be installed._

```sh
cd contributor-1 && poetry install && cd -
cd contributor-2 && poetry install && cd -
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

Use any port number `{8001,8002}`:

```sh
# visit in any browser
http://localhost:<PORT>/docs#/
```

## Building and Running the NetworkRetriever

Let's create the kernel for our notebook so that we can use our newly
created `networks-retriever-demo` virtual environment. With the `networks-retriever-demo`
virtual environment still active, run the below shell command:

```sh
ipython kernel install --user --name=networks-retriever-demo
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

From there you can open up the `network_retriever.ipynb`.
