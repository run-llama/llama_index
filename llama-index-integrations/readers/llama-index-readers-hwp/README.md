# HWP Loader

```bash
pip install llama-index-readers-file
```

This loader reads the HWP file, which is the format of many official documents in South Korea.

## Usage

To use this loader, you need to pass in a file name. It's fine whether the file is compressed or not.

```python
from llama_index.readers.file import HWPReader
from pathlib import Path

hwp_path = Path("/path/to/hwp")
reader = HWPReader()
documents = reader.load_data(file=hwp_path)
```
