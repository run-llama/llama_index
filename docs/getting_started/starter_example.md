# Starter Tutorial

Here is a starter example for using GPT Index. Make sure you've followed the [installation](installation.md) steps first.


### Download
GPT Index examples can be found in the `examples` folder of the GPT Index repository. 
We first want to download this `examples` folder. An easy way to do this is to just clone the repo: 
`git clone git@github.com:jerryjliu/gpt_index.git`.


We now want to navigate to the following folder:
```bash
cd examples/paul_graham_essay
```

This contains GPT Index examples around Paul Graham's essay, ["What I Worked On"](http://paulgraham.com/worked.html). A comprehensive set of examples are already provided in `TestEssay.ipynb`. For the purposes of this tutorial, we can focus on a simple example of getting GPT Index up and running.


### Build and Query Index
Create a new `.py` file with the following:

```python
from gpt_index import GPTTreeIndex, SimpleDirectoryReader
from IPython.display import Markdown, display

documents = SimpleDirectoryReader('data').load_data()
index = GPTTreeIndex(documents)

```

This builds an index over the documents in the `data` folder (which in this case just consists of the essay text). We then run the following
```python
response = index.query("What did the author do growing up?")
```

You should get back a response similar to the following: `The author wrote short stories and tried to program on an IBM 1401.`

### Saving and Loading

To save to disk and load from disk, do

```python
# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTTreeIndex.load_from_disk('index.json')
```


### Next Steps

That's it! 
TODO: add next steps

