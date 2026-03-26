# Open Weather Map Tool

This tool connects to the [OpenWeatherMap](https://openweathermap.org/api)'s OneCall API, using the `pyowm` Python package. You must initialize the tool with your OpenWeatherMap API token

The tool has access to the following functions:

- the current weather
- the weather tomorrow

## Usage

Here's an example usage of the OpenWeatherMapToolSpec.

```python
from llama_index.tools.weather import OpenWeatherMapToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = OpenWeatherMapToolSpec(key="...")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("What is the temperature like in Paris?"))
print(await agent.run("What is the wind like in Budapest tomorrow?"))
```

`weather_at_location`: Use pyowm to get current weather details at a location

`forecast_tomorrow_at_location`: Use pyowm to get the forecast for tomorrow at a location.
