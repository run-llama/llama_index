Command line interface (experimental)
========

This module providers a way to interactive with llama\_index directly in shell.

## Get started
Because "experimental" is not included in the package yet (I think it's why it called "experimental"). For now, you need to git clone this repo and run these command in it.
Or you can set `export PYTHONPATH=/path/to/your/llama_index` before run following command.

For long term, when this part of code is stable enough, we can move to src. At that time user will be able to call it directly with something like `python -m llama_index.cli init`.

### Commands

```
python -m experimental.cli init
```
this creates a config file "config.ini". If default config is not enough for you, you need to update it manually (for now).

```
python -m experimental.cli add ../data/
```
This use OpenAI/Azure OpenAI API to "analysis" files under "../data/" and store it in a file named "index.json" in current directory.

```
python -m experimental.cli query "Some question?"
```
This checks the local "index.json" and send some more query to OpenAI/Azure OpenAI for the answer to your question.

There're two files put in current directory.

- config.ini stores embedding/predicter model setup along with its parameters
- index.json the index file

## Configuration

### Index

Support following type of index:

#### Vector index
```ini
[index]
type = vector
```

#### Key word index
```ini
[index]
type = keyword
```

### Embedding

Support following type of embedding:

#### Default
```ini
[embed_model]
type = default
```

### LLM predictor

Support following type of LLM predictor:

#### Default
```ini
[llm_predictor]
type = default
```
If you're using Azure OpenAI API, add `engine`:

```ini
[llm_predictor]
type = default
engine = text-davinci-003
```

#### Structured LLM
```ini
[llm_predictor]
type = structured
```
It also supports Azure OpenAI API `engine`.

## Examples

#### Default setup
```ini
[store]
type = json

[index]
type = default

[embed_model]
type = default

[llm_predictor]
type = default
```

#### Keyword + structured
```ini
[store]
type = json

[index]
type = keyword

[llm_predictor]
type = structured
engine = text-davinci-003
```
