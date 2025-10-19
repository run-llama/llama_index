# Migration Guide: SVG Support Changes

## Overview

Starting from version 0.4.5, SVG processing support has been moved to an optional dependency to address installation issues on systems where the `pycairo` package cannot be compiled (particularly Debian/Ubuntu systems without C compilers or Cairo development libraries).

## What Changed?

### Before (versions < 0.4.5)

- `svglib` was a required dependency
- All users had to install `pycairo` even if they didn't need SVG support
- Installation could fail on systems without proper build tools

### After (versions >= 0.4.5)

- `svglib` is now an optional dependency
- SVG processing is skipped by default with a warning if optional dependencies are not installed
- Base installation works on all systems without requiring C compilers
- SVG version pinned to `<1.6.0` to avoid breaking changes

## Migration Paths

### Option 1: Continue Using Built-in SVG Support (Recommended if SVG is needed)

If you need SVG processing and can install the required system dependencies:

```bash
# Uninstall current version
pip uninstall llama-index-readers-confluence

# Install with SVG support
pip install 'llama-index-readers-confluence[svg]'
```

**System Requirements for SVG Support:**

- On Debian/Ubuntu: `sudo apt-get install gcc python3-dev libcairo2-dev`
- On macOS: `brew install cairo`
- On Windows: Install Visual C++ Build Tools

### Option 2: Skip SVG Processing (Recommended for Docker/CI environments)

If you don't need SVG processing or want to avoid installation issues:

```bash
# Install without SVG support (default)
pip install llama-index-readers-confluence
```

SVG attachments will be skipped with a warning in the logs. All other functionality remains unchanged.

### Option 3: Use Custom SVG Parser

If you need SVG processing but cannot install pycairo, use a custom parser:

```python
from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import FileType


# Simple text extraction from SVG (no OCR)
class SimpleSVGParser(BaseReader):
    def load_data(self, file_path, **kwargs):
        import xml.etree.ElementTree as ET

        with open(file_path, "r") as f:
            root = ET.fromstring(f.read())

        # Extract text elements from SVG
        texts = [elem.text for elem in root.findall(".//text") if elem.text]
        extracted_text = " ".join(texts) or "[SVG Image]"

        return [
            Document(text=extracted_text, metadata={"file_path": file_path})
        ]


reader = ConfluenceReader(
    base_url="https://yoursite.atlassian.com/wiki",
    api_token="your_token",
    custom_parsers={FileType.SVG: SimpleSVGParser()},
)
```

See `examples/svg_parsing_examples.py` for more custom parser examples.

### Option 4: Filter Out SVG Attachments

If you want to explicitly skip SVG files without warnings:

```python
def attachment_filter(
    media_type: str, file_size: int, title: str
) -> tuple[bool, str]:
    if media_type == "image/svg+xml":
        return False, "SVG processing disabled"
    return True, ""


reader = ConfluenceReader(
    base_url="https://yoursite.atlassian.com/wiki",
    api_token="your_token",
    process_attachment_callback=attachment_filter,
)
```

## Docker/Container Deployments

### Before (versions < 0.4.5)

```dockerfile
FROM python:3.11-slim

# Required system dependencies for pycairo
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install llama-index-readers-confluence
```

### After (versions >= 0.4.5) - Without SVG Support

```dockerfile
FROM python:3.11-slim

# No system dependencies needed!
RUN pip install llama-index-readers-confluence
```

### After (versions >= 0.4.5) - With SVG Support

```dockerfile
FROM python:3.11-slim

# Only if you need SVG support
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install 'llama-index-readers-confluence[svg]'
```

## FAQ

### Q: Will my existing code break?

**A:** No, your existing code will continue to work. If you were using SVG processing and don't install the `[svg]` extra, SVG attachments will simply be skipped with a warning instead of failing.

### Q: How do I know if SVG dependencies are installed?

**A:** Check the logs. If you see warnings like "SVG processing skipped: Optional dependencies not installed", then SVG dependencies are not available.

### Q: Can I use a different OCR engine for SVG?

**A:** Yes! Use the custom parser approach (Option 3) and implement your own SVG-to-text conversion logic. You could use libraries like `cairosvg`, `pdf2image`, or pure XML parsing depending on your needs.

### Q: Why was this change made?

**A:** The `pycairo` dependency (required by `svglib`) requires C compilation and system libraries (Cairo). This caused installation failures in:

- Docker containers based on slim images
- CI/CD pipelines without build tools
- Systems managed by users without admin rights
- Environments where SVG support isn't needed

Making it optional allows the package to work everywhere while still supporting SVG for users who need it.

### Q: What if I encounter other issues?

**A:** Please file an issue on GitHub with:

1. Your Python version
2. Your operating system
3. Whether you installed with `[svg]` extra
4. The full error message
5. Output of `pip list` showing installed packages

## Testing Your Migration

After migrating, test your setup:

```python
from llama_index.readers.confluence import ConfluenceReader
import logging

# Enable logging to see SVG warnings
logging.basicConfig(level=logging.INFO)

reader = ConfluenceReader(
    base_url="https://yoursite.atlassian.com/wiki",
    api_token="your_token",
)

# Try loading data
documents = reader.load_data(space_key="MYSPACE", include_attachments=True)

# Check logs for any SVG-related warnings
print(f"Loaded {len(documents)} documents")
```

If you see "SVG processing skipped" warnings but didn't expect them, you may need to install the `[svg]` extra.
