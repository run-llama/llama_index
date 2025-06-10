# LlamaIndex Utils: Workflow

Utilities for LlamaIndex workflows, including visualization tools.

## Features

- **Workflow visualization** with `draw_all_possible_flows()`
- **Label truncation support** for better readability with long event names

## Usage

```python
from llama_index.utils.workflow import draw_all_possible_flows

# Basic workflow visualization
draw_all_possible_flows(my_workflow, "workflow.html")

# With label truncation for long event names (v0.4.0+)
draw_all_possible_flows(my_workflow, "workflow.html", max_label_length=15)
```
