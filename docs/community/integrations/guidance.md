
# Guidance

[Guidance](https://github.com/microsoft/guidance) is a guidance language for controlling large language models developed by Microsoft.

Guidance programs allow you to interleave generation, prompting, and logical control into a single continuous flow matching how the language model actually processes the text.

## Structured Output
One particularly exciting aspect of guidance is the ability to output structured objects (think JSON following a specific schema, or a pydantic object). Instead of just "suggesting" the desired output structure to the LLM, guidance can actually "force" the LLM output to follow the desired schema. This allows the LLM to focus on the content rather than the syntax, and completely eliminate the possibility of output parsing issues.

This is particularly powerful for weaker LLMs which be smaller in parameter count, and not trained on sufficient source code data to be able to reliably produce well-formed, hierarchical structured output.

### Creating a guidance program to generate pydantic objects 
In LlamaIndex, we provide an initial integration with guidance, to make it super easy for generating structured output (more specifically pydantic objects).

For example, if we want to generate an album of songs, with the following schema:

```python
class Song(BaseModel):
    title: str
    length_seconds: int
    
class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]
```

It's as simple as creating a `GuidancePydanticProgram`, specifying our desired pydantic class `Album`, 
and supplying a suitable prompt template.

> Note: guidance uses handlebars-style templates, which uses double braces for variable substitution, and single braces for literal braces. This is the opposite convention of Python format strings. 

> Note: We provide an utility function `from llama_index.prompts.guidance_utils import convert_to_handlebars` that can convert from the Python format string style template to guidance handlebars-style template.


```python
program = GuidancePydanticProgram(
    output_cls=Album, 
    prompt_template_str="Generate an example album, with an artist and a list of songs. Using the movie {{movie_name}} as inspiration",
    guidance_llm=OpenAI('text-davinci-003'),
    verbose=True,
)

```

Now we can run the program by calling it with additional user input. 
Here let's go for something spooky and create an album inspired by the Shining.
```python
output = program(movie_name='The Shining')
```

We have our pydantic object:
```python
Album(name='The Shining', artist='Jack Torrance', songs=[Song(title='All Work and No Play', length_seconds=180), Song(title='The Overlook Hotel', length_seconds=240), Song(title='The Shining', length_seconds=210)])
```

You can play with [this notebook](/examples/output_parsing/guidance_pydantic_program.ipynb) for more details.

### Using guidance to improve the robustness of our sub-question query engine.
LlamaIndex provides a toolkit of advanced query engines for tackling different use-cases.
Several relies on structured output in intermediate steps.
We can use guidance to improve the robustness of these query engines, by making sure the
intermediate response has the expected structure (so that they can be parsed correctly to a structured object).

As an example, we implement a `GuidanceQuestionGenerator` that can be plugged into a `SubQuestionQueryEngine` to make it more robust than using the default setting.
```python
from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator
from guidance.llms import OpenAI as GuidanceOpenAI

# define guidance based question generator
question_gen = GuidanceQuestionGenerator.from_defaults(guidance_llm=GuidanceOpenAI('text-davinci-003'), verbose=False)

# define query engine tools
query_engine_tools = ...

# construct sub-question query engine
s_engine = SubQuestionQueryEngine.from_defaults(
    question_gen=question_gen  # use guidance based question_gen defined above
    query_engine_tools=query_engine_tools, 
)
```

See [this notebook](/examples/output_parsing/guidance_sub_question.ipynb) for more details.






