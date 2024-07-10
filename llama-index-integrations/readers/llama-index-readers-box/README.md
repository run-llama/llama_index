# Box File or Directory Loader

This loader parses any file stored on Box, or the entire folder (with an optional recursive option) if no particular file is specified. When initializing `BoxReader`, you can use Developer Token authentication method.

All files are parsed with `SimpleDirectoryReader`. You may also specify a custom `file_extractor`, relying on any of the loaders in this library (or your own)!

## Installation

```bash
pip install llama-index-readers-box
```

## Usage

To use this loader, you need to authenticate using Developer Token. After that, if you want to parse a single file, pass in its file ID. Otherwise, you can specify a folder ID to parse all files in the folder.

### Using Developer Token

```python
loader = BoxReader(
    auth_method="developer_token",
    developer_token="[DEVELOPER_TOKEN]",
    folder_id="0",  # Root folder
    recursive=True,
    num_files_limit=100,
)
documents = loader.load_data()
```

### Extra Parameters in BoxReader

The `BoxReader` class includes several parameters that provide additional customization and control over the data loading process. Here’s an explanation of these parameters:

1. **`auth_method`**: Specifies the authentication method to use. It can be `"developer_token"`, in future more methods will be added. This parameter determines how the Box API is accessed.

   - **Possible values**: `"developer_token"`
   - **Default value**: `None` (must be explicitly set)

2. **Developer Token Parameter**:

   - **`developer_token`**: The developer token for authentication. Required if `auth_method` is `"developer_token"`.

3. **`folder_id`**: The ID of the folder to parse. If not specified, the root folder (`"0"`) is used.

   - **Default value**: `None` (defaults to the root folder)

4. **`recursive`**: Indicates whether to parse folders recursively. If set to `True`, the `BoxReader` will navigate through subfolders.

   - **Default value**: `True`

5. **`file_extractor`**: A custom file extractor to use for processing files. This can be a dictionary specifying how different file types should be processed.

   - **Default value**: `None`

6. **`num_files_limit`**: The maximum number of files to parse. This is useful for limiting the number of files processed, especially when working with large directories.

   - **Default value**: `None` (no limit)

7. **`chunk_size`**: The chunk size for streaming large files. This controls how much data is read in each chunk during the streaming of large files.
   - **Default value**: `5 * 1024 * 1024` (5 MB)

8 **`large_file_threshold`**: The file size threshold for treating a file as large. Files exceeding this size will be streamed in chunks instead of being read all at once. - **Default value**: `10 * 1024 * 1024` (10 MB)

### Usage Examples

Here are some examples of how to use these extra parameters:

#### Specifying a Custom File Extractor

You can specify a custom file extractor to define how different file types should be processed:

```python
file_extractor = {
    ".txt": lambda x: x.read().decode("utf-8"),
    ".pdf": custom_pdf_extractor,
}

loader = BoxReader(
    auth_method="developer_token",
    developer_token="[DEVELOPER_TOKEN]",
    recursive=True,
    file_extractor=file_extractor,
)
documents = loader.load_data()
```

#### Limiting the Number of Files

You can limit the number of files processed by setting `num_files_limit`:

```python
loader = BoxReader(
    auth_method="developer_token",
    developer_token="[DEVELOPER_TOKEN]",
    recursive=True,
    num_files_limit=50,  # Process only the first 50 files
)
documents = loader.load_data()
```

#### Adjusting the Chunk Size and Large File Threshold

You can adjust the chunk size for streaming and the threshold for considering a file as large:

```python
loader = BoxReader(
    auth_method="developer_token",
    developer_token="[DEVELOPER_TOKEN]",
    recursive=True,
    chunk_size=2 * 1024 * 1024,  # 2 MB chunk size
    large_file_threshold=20 * 1024 * 1024,  # 20 MB threshold
)
documents = loader.load_data()
```

By understanding and utilizing these extra parameters, you can tailor the `BoxReader` to better fit your specific needs and use cases.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
