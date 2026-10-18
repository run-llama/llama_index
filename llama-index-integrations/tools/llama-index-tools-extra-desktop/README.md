# LlamaIndex Extra Desktop Tool

This tool integration provides autonomous desktop computer-use and GUI automation capabilities for LlamaIndex agents across Linux, Windows, and macOS, powered by [Project Extra](https://github.com/AIYantra/extra).

It enables agents to interact directly with desktop operating systems and native GUI applications with sub-3ms perception, live Bezier mouse movements, instant Win32/uinput typing, window management, and compound macro workflows.

## Installation

```bash
pip install llama-index-tools-extra-desktop extra-desktop
```

## Usage

```python
from llama_index.tools.extra_desktop import ExtraDesktopToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Initialize Extra Desktop Tool Spec
desktop_spec = ExtraDesktopToolSpec()
tools = desktop_spec.to_tool_list()

# Initialize LLM & Agent
llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# Run a desktop task
agent.chat("Launch the desktop calculator and calculate 25 * 40.")
```

## Available Functions

- `launch_app(app_name, args)`: Launch a desktop application or process.
- `click(target, x, y, button)`: Click on a UI element via semantic grounding or coordinates.
- `type_text(text, press_enter)`: Type text into the active window.
- `send_hotkey(keys)`: Send global keyboard shortcuts (e.g. `['ctrl', 's']`).
- `capture_screen()`: Capture the desktop screen.
- `snap_layout(layout, left_window, right_window)`: Snap windows side-by-side.
- `batch_actions(actions)`: Execute a compound sequence of hardware actions in a single rapid batch.
- `run_subtask(intent)`: Execute a closed-loop macro intent.
