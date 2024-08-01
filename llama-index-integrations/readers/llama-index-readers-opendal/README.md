# OpenDAL Loaders

```bash
pip install llama-index-readers-opendal
```

## Base OpendalReader

This loader parses any file via [Apache OpenDAL](https://github.com/apache/incubator-opendal).

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

### Usage

`OpendalReader` can read data from any supported storage services including `s3`, `azblob`, `gcs` and so on.

```python
from llama_index.readers.opendal import OpendalReader

loader = OpendalReader(
    scheme="s3",
    bucket="bucket",
    path="path/to/data/",
)
documents = loader.load_data()
```

We also provide `Opendal[S3|Gcs|Azblob]Reader` for convenience.

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

## Azblob Loader

This loader parses any file stored on Azblob.

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

> Azblob loader is based on `OpendalReader`.

### Usage

```python
from llama_index.readers.opendal import OpendalAzblobReader

loader = OpendalAzblobReader(
    container="container",
    path="path/to/data/",
    endpoint="[endpoint]",
    account_name="[account_name]",
    account_key="[account_key]",
)
documents = loader.load_data()
```

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

## Gcs Loader

This loader parses any file stored on Gcs.

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

> Gcs loader is based on `OpendalReader`.

### Usage

```python
from llama_index.readers.opendal import OpendalGcsReader

loader = OpendalGcsReader(
    bucket="bucket",
    path="path/to/data/",
    endpoint="[endpoint]",
    credentials="[credentials]",
)
documents = loader.load_data()
```

Note: if `credentials` is not provided, this loader to try to load from env.

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

## S3 Loader

This loader parses any file stored on S3. When initializing `S3Reader`, you may pass in your [AWS Access Key](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html). If none are found, the loader assumes they are stored in `~/.aws/credentials`.

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

> S3 loader is based on `OpendalReader`.

### Usage

```python
loader = OpendalS3Reader(
    bucket="bucket",
    path="path/to/data/",
    access_key_id="[ACCESS_KEY_ID]",
    secret_access_key="[ACCESS_KEY_SECRET]",
)
documents = loader.load_data()
```

Note: if `access_key_id` or `secret_access_key` is not provided, this loader to try to load from env.

Possible arguments includes:

- `endpoint`: Specify the endpoint of s3 service.
- `region`: Specify the region of s3 service.

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
