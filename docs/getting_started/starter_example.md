# Starter Tutorial

Here is a starter example for using GPT Index. Make sure you've followed the [installation](installation.md) steps first.

### Download

GPT Index examples can be found in the `examples` folder of the GPT Index repository.
We first want to download this `examples` folder. An easy way to do this is to just clone the repo:

```bash
$ git clone https://github.com/jerryjliu/gpt_index.git
```

Next, navigate to your newly-cloned repository, and verify the contents:

```bash
$ cd gpt_index
$ ls
LICENSE                data_requirements.txt  tests/
MANIFEST.in            examples/              pyproject.toml
Makefile               experimental/          requirements.txt
README.md              gpt_index/             setup.py
```

We now want to navigate to the following folder:

```bash
$ cd examples/paul_graham_essay
```

This contains GPT Index examples around Paul Graham's essay, ["What I Worked On"](http://paulgraham.com/worked.html). A comprehensive set of examples are already provided in `TestEssay.ipynb`. For the purposes of this tutorial, we can focus on a simple example of getting GPT Index up and running.

### Build and Query Index

Create a new `.py` file with the following:

```python
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(documents)
```

This builds an index over the documents in the `data` folder (which in this case just consists of the essay text). We then run the following

```python
response = index.query("What did the author do growing up?")
print(response)
```

You should get back a response similar to the following: `The author wrote short stories and tried to program on an IBM 1401.`

### Viewing Queries and Events Using Logging

In a Jupyter notebook, you can view info and/or debugging logging using the following snippet:

```python
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

You can set the level to `DEBUG` for verbose output, or use `level=logging.INFO` for less.

### Saving and Loading

To save to disk and load from disk, do

```python
# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')
```

### Next Steps

That's it! For more information on GPT Index features, please check out the numerous "How-To Guides" to the left.
Additionally, if you would like to play around with Example Notebooks, check out [this link](/reference/example_notebooks.rst).
