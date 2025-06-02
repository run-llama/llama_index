# Image Tabular Chart Loader (Deplot)

```bash
pip install llama-index-readers-file
```

This loader captions an image file containing a tabular chart (bar chart, line charts) using deplot.

## Usage

To use this loader, you need to pass in a `Path` to a local file.

```python
from pathlib import Path
from llama_index.readers.file import ImageTabularChartReader

loader = ImageTabularChartReader()
documents = loader.load_data(file=Path("./image.png"))
```
