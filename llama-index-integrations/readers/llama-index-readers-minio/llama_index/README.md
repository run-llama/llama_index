# Minio File or Directory Loader

## Boto

This loader parses any file stored on Minio, or the entire Bucket (with an optional prefix filter) if no particular file is specified. When initializing `BotoMinioReader`, you may pass in your `minio_access_key` and `minio_secret_key` as `aws_access_id` and `aws_access_secret` respectively.

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

## Usage

To use this loader, you need to pass in the name of your Minio Bucket. After that, if you want to just parse a single file, pass in its key. Note that if the file is nested in a subdirectory, the key should contain that, so like `subdirectory/input.txt`.

Otherwise, you may specify a prefix if you only want to parse certain files in the Bucket, or a subdirectory.

```python
MinioReader = download_loader("BotoMinioReader")
loader = MinioReader(
    bucket="documents",
    aws_access_id="minio_access_key",
    aws_access_secret="minio_secret_key",
    s3_endpoint_url="localhost:9000",
)
documents = loader.load_data()
```

## Minio File

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

All files are temporarily downloaded locally and subsequently parsed with `SimpleDirectoryReader`. Hence, you may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

## Usage

To use this loader, you need to pass in the name of your Minio Bucket. After that, if you want to just parse a single file, pass in its key. Note that if the file is nested in a subdirectory, the key should contain that, so like `subdirectory/input.txt`.

Otherwise, you may specify a prefix if you only want to parse certain files in the Bucket, or a subdirectory.

You can now use the client with a TLS-secured MinIO instance (`minio_secure=True`), even if server's certificate isn't trusted (`minio_cert_check=False`).

```python
MinioReader = download_loader("MinioReader")
loader = MinioReader(
    bucket="documents",
    minio_endpoint="localhost:9000",
    minio_secure=True,
    minio_cert_check=False,
    minio_access_key="minio_access_key",
    minio_secret_key="minio_secret_key",
)
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
