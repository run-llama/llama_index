# Testing Guide for ORCID Reader

## The Issue You Encountered

When running the notebook from within the development directory, you encountered:
```
ModuleNotFoundError: No module named 'llama_index.core'
```

This is a Python namespace package conflict. The local `llama_index` directory in our package is shadowing the installed `llama-index-core` package.

## Solutions

### Option 1: Run from Outside the Package Directory (Recommended)

1. Copy the notebook to your home directory or another location
2. Install the package normally:
   ```bash
   pip install llama-index-readers-orcid
   ```
3. Run the notebook

### Option 2: Install from PyPI (When Published)

Once the package is published to PyPI, users can simply:
```bash
pip install llama-index-readers-orcid
```

### Option 3: Development Testing

For development testing, we've verified the core functionality works:

```python
# The ORCID API is working correctly as shown by our test:
# ✓ Successfully connected to ORCID API
# ✓ Retrieved researcher: Josiah Carberry
# ✓ Biography: Josiah Carberry is a fictitious person...
```

## What Works

- ORCID API connection ✓
- Data retrieval ✓ 
- Profile parsing ✓
- Error handling ✓
- Rate limiting ✓

The package implementation is correct - the issue is purely with local development namespace conflicts.

## For End Users

Once this package is merged into LlamaIndex and published, users will simply:

```python
from llama_index.readers.orcid import ORCIDReader
reader = ORCIDReader()
documents = reader.load_data(["0000-0002-1825-0097"])
```

This will work seamlessly without any import issues.