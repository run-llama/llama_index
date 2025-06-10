# LlamaIndex Utils: Workflow

Utilities for LlamaIndex workflows, including visualization tools.

```bash
pip install llama-index-utils-workflow
```

## Features

- **Workflow visualization** with `draw_all_possible_flows()`
- **Latest execution visualization** with `draw_most_recent_execution()`
- **Label truncation support** for better readability with long event names

## Usage

```python
from llama_index.utils.workflow import draw_all_possible_flows

# Basic workflow visualization
draw_all_possible_flows(my_workflow, "workflow.html")

# With label truncation for long event names (v0.4.0+)
draw_all_possible_flows(my_workflow, "workflow.html", max_label_length=15)

# Latest execution visualization
result = await my_workflow.run()
draw_most_recent_execution(my_workflow, "workflow.html")
```
