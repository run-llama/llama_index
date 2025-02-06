# SimpleDirectoryReader

`SimpleDirectoryReader` is the simplest way to load data from local files into LlamaIndex. For production use cases it's more likely that you'll want to use one of the many Readers available on [LlamaHub](https://llamahub.ai/), but `SimpleDirectoryReader` is a great way to get started.

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

One file type you may be expecting to find here is JSON; for that we recommend you use our [JSON Loader](https://llamahub.ai/l/readers/llama-index-readers-json).

## Usage

The most basic usage is to pass an `input_dir` and it will load all supported files in that directory:

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

Documents can also be loaded with parallel processing if loading many files from
a directory. Note that there are differences when using `multiprocessing` with
Windows and Linux/MacOS machines, which is explained throughout the `multiprocessing` docs
(e.g. see [here](https://docs.python.org/3/library/multiprocessing.html?highlight=process#the-spawn-and-forkserver-start-methods)).
Ultimately, Windows users may see less or no performance gains whereas Linux/MacOS
users would see these gains when loading the exact same set of files.

```python
...
documents = reader.load_data(num_workers=4)
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

`SimpleDirectoryReader` will automatically attach a `metadata` dictionary to each `Document` object. By default, this dictionary has these items:

- `file_path`: the full filesystem path to the file, including the file name (string)
- `file_name`: the file name, including suffix (string)
- `file_type`: the MIME type of the file, as guessed by [`mimetypes.guess_type()](https://docs.python.org/3/library/mimetypes.html#mimetypes.guess_type) (string)
- `file_size`: the size of the file, in bytes (integer)
- `creation_date`, `last_modified_date`, `last_accessed_date`: the creation, modification, and access dates for the file, normalized to the UTC timezone. See [Date and time metadata](#date-and-time-metadata) below (string)

However, you can replace the logic used to create the metadata dictionary. Create a custom function which takes a file path string and returns a dictionary, then pass this function to the `SimpleDirectoryReader` constructor as `file_metadata`:

```python
def get_meta(file_path):
    return {"foo": "bar", "file_path": file_path}


reader = SimpleDirectoryReader(
    input_dir="path/to/directory", file_metadata=get_meta
)

docs = reader.load_data()
print(docs[0].metadata["foo"])  # prints "bar"
```

#### Date and time metadata

The default metadata function in `SimpleDirectoryReader` outputs dates as a string with the [format](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) `%Y-%m-%d`.

To ensure consistency, timestamps are normalized to the UTC timezone. If the output dates seem to be one day off from the real date, this may be explained by the offset with midnight UTC.

### Extending to other file types

You can extend `SimpleDirectoryReader` to read other file types by passing a dictionary of file extensions to instances of `BaseReader` as `file_extractor`. A BaseReader should read the file and return a list of Documents. For example, to add custom support for `.myfile` files :

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document


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

### Support for External FileSystems

As with other modules, the `SimpleDirectoryReader` takes an optional `fs` parameter that can be used to traverse remote filesystems.

This can be any filesystem object that is implemented by the [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) protocol.
The `fsspec` protocol has open-source implementations for a variety of remote filesystems including [AWS S3](https://github.com/fsspec/s3fs), [Azure Blob & DataLake](https://github.com/fsspec/adlfs), [Google Drive](https://github.com/fsspec/gdrivefs), [SFTP](https://github.com/fsspec/sshfs), and [many others](https://github.com/fsspec/).

Here's an example that connects to S3:

```python
from s3fs import S3FileSystem

s3_fs = S3FileSystem(key="...", secret="...")
bucket_name = "my-document-bucket"

reader = SimpleDirectoryReader(
    input_dir=bucket_name,
    fs=s3_fs,
    recursive=True,  # recursively searches all subdirectories
)

documents = reader.load_data()
print(documents)
```

A full example notebook can be found [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/simple_directory_reader_remote_fs.ipynb).
