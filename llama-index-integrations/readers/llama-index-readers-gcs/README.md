# GCS File or Directory Loader

This loader parses any file stored on GCS, or the entire Bucket (with an optional prefix filter) if no particular file is specified. When initializing `GCSReader`, you may pass in your [GCP Service Account Key](https://cloud.google.com/iam/docs/keys-create-delete). If none are found, the loader assumes they are stored in `~/.gcp/credentials`.

All files are parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

## Usage

To use this loader, you need to pass in the name of your GCS Bucket. After that, if you want to just parse a single file, pass in its key. Note that if the file is nested in a subdirectory, the key should contain that, so like `subdirectory/input.txt`.

Otherwise, you may specify a prefix if you only want to parse certain files in the Bucket, or a subdirectory. GCP Service Account Key credentials may either be passed in during initialization or stored locally (see above).

```python
loader = GCSReader(
    bucket="scrabble-dictionary",
    key="dictionary.txt",
    service_account_key_json="[SERVICE_ACCOUNT_KEY_JSON]",
)
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
