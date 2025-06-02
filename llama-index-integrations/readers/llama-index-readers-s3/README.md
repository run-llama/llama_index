# S3 File or Directory Loader

This loader parses any file stored on S3, or the entire Bucket (with an optional prefix filter) if no particular file is specified. When initializing `S3Reader`, you may pass in your [AWS Access Key](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html). If none are found, the loader assumes they are stored in `~/.aws/credentials`.

All files are parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

## Installation

```bash
pip install llama-index-readers-s3
```

## Usage

To use this loader, you need to pass in the name of your S3 Bucket. After that, if you want to just parse a single file, pass in its key. Note that if the file is nested in a subdirectory, the key should contain that, so like `subdirectory/input.txt`.

Otherwise, you may specify a prefix if you only want to parse certain files in the Bucket, or a subdirectory. AWS Access Key credentials may either be passed in during initialization or stored locally (see above).

```python
loader = S3Reader(
    bucket="scrabble-dictionary",
    key="dictionary.txt",
    aws_access_id="[ACCESS_KEY_ID]",
    aws_access_secret="[ACCESS_KEY_SECRET]",
)
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
