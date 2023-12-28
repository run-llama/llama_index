# SimpleDirectoryReader

`SimpleDirectoryReader` is the simplest way to load data from local files into LlamaIndex. For production use cases it's more likely that you'll want to use one of the many Readers available on [LlamaHub](https://llamalab.com/hub), but `SimpleDirectoryReader` is a great way to get started.

## Supported file types

By default `SimpleDirectoryReader` will try to read any files it finds, treating them all as text. In addition to plain text, it explicitly supports the following file types, which are automatically detected based on file extension:

- .csv - comma-separated values
- .docx - Microsoft Word
- .epub - EPUB ebook format
- .hwp - Hangul Word Processor
- .ipynb - Jupyter Notebook
- .jpeg, .jpg - JPEG image
- .mbox - MBOX email archive
- .md - Markdown
- .mp3, .mp4 - audio and video
- .pdf - Portable Document Format
- .png - Portable Network Graphics
- .ppt, .pptm, .pptx - Microsoft PowerPoint

One file type you may be expecting to find here is JSON; for that we recommend you use our [JSON Loader](https://llamahub.ai/l/file-json).

## Usage

The most basic usage is to pass an `input_dir` and it will load all supported files in that directory:

```python
from llama_index import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

### Reading from subdirectories

By default, `SimpleDirectoryReader` will only read files in the top level of the directory. To read from subdirectories, set `recursive=True`:

```python
SimpleDirectoryReader(input_dir="path/to/directory", recursive=True)
```

### Iterating over files as they load

You can also use the `iter_data()` method to iterate over and process files as they load

```python
reader = SimpleDirectoryReader(input_dir="path/to/directory", recursive=True)
all_docs = []
for docs in reader.iter_data():
    # <do something with the documents per file>
    all_docs.extend(docs)
```

### Restricting the files loaded

Instead of all files you can pass a list of file paths:

```python
SimpleDirectoryReader(input_files=["path/to/file1", "path/to/file2"])
```

or you can pass a list of file paths to **exclude** using `exclude`:

```python
SimpleDirectoryReader(
    input_dir="path/to/directory", exclude=["path/to/file1", "path/to/file2"]
)
```

You can also set `required_exts` to a list of file extensions to only load files with those extensions:

```python
SimpleDirectoryReader(
    input_dir="path/to/directory", required_exts=[".pdf", ".docx"]
)
```

And you can set a maximum number of files to be loaded with `num_files_limit`:

```python
SimpleDirectoryReader(input_dir="path/to/directory", num_files_limit=100)
```

### Specifying file encoding

`SimpleDirectoryReader` expects files to be `utf-8` encoded but you can override this using the `encoding` parameter:

```python
SimpleDirectoryReader(input_dir="path/to/directory", encoding="latin-1")
```

### Extracting metadata

You can specify a function that will read each file and extract metadata that gets attached to the resulting `Document` object for each file by passing the function as `file_metadata`:

```python
def get_meta(file_path):
    return {"foo": "bar", "file_path": file_path}


SimpleDirectoryReader(input_dir="path/to/directory", file_metadata=get_meta)
```

The function should take a single argument, the file path, and return a dictionary of metadata.

### Extending to other file types

You can extend `SimpleDirectoryReader` to read other file types by passing a dictionary of file extensions to instances of `BaseReader` as `file_extractor`. A BaseReader should read the file and return a list of Documents. For example, to add custom support for `.myfile` files :

```python
from llama_index import SimpleDirectoryReader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text + "Foobar", extra_info=extra_info or {})]


reader = SimpleDirectoryReader(
    input_dir="./data", file_extractor={".myfile": MyFileReader()}
)

documents = reader.load_data()
print(documents)
```

Note that this mapping will override the default file extractors for the file types you specify, so you'll need to add them back in if you want to support them.
