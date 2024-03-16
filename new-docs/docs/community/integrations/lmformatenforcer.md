# LM Format Enforcer

[LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) is a library that enforces the output format (JSON Schema, Regex etc) of a language model. Instead of just "suggesting" the desired output structure to the LLM, LM Format Enforcer can actually "force" the LLM output to follow the desired schema.

![image](https://raw.githubusercontent.com/noamgat/lm-format-enforcer/main/docs/Intro.webp)

LM Format Enforcer works with local LLMs (currently supports `LlamaCPP` and `HuggingfaceLLM` backends), and operates only by processing the output logits of the LLM. This enables it to support advanced generation methods like beam search and batching, unlike other solutions that modify the generation loop itself. See the comparison table in the [LM Format Enforcer page](https://github.com/noamgat/lm-format-enforcer) for more details.

## JSON Schema Output

In LlamaIndex, we provide an initial integration with LM Format Enforcer, to make it super easy for generating structured output (more specifically pydantic objects).

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

It's as simple as creating a `LMFormatEnforcerPydanticProgram`, specifying our desired pydantic class `Album`,
and supplying a suitable prompt template.

> Note: `LMFormatEnforcerPydanticProgram` automatically fills in the json schema of the pydantic class in the optional `{json_schema}` parameter of the prompt template. This can help the LLM naturally generate the correct JSON and reduce the interference aggression of the format enforcer, increasing output quality.

```python
program = LMFormatEnforcerPydanticProgram(
    output_cls=Album,
    prompt_template_str="Generate an example album, with an artist and a list of songs. Using the movie {movie_name} as inspiration. You must answer according to the following schema: \n{json_schema}\n",
    llm=LlamaCPP(),
    verbose=True,
)
```

Now we can run the program by calling it with additional user input.
Here let's go for something spooky and create an album inspired by the Shining.

```python
output = program(movie_name="The Shining")
```

We have our pydantic object:

```python
Album(
    name="The Shining: A Musical Journey Through the Haunted Halls of the Overlook Hotel",
    artist="The Shining Choir",
    songs=[
        Song(title="Redrum", length_seconds=300),
        Song(
            title="All Work and No Play Makes Jack a Dull Boy",
            length_seconds=240,
        ),
        Song(title="Heeeeere's Johnny!", length_seconds=180),
    ],
)
```

You can play with [this notebook](../../examples/output_parsing/lmformatenforcer_pydantic_program.ipynb) for more details.

## Regular Expression Output

LM Format Enforcer also supports regex output. Since there is no existing abstraction for regular expressions in LlamaIndex, we will use the LLM directly, after injecting the LM Format Generator in it.

```python
regex = r'"Hello, my name is (?P<name>[a-zA-Z]*)\. I was born in (?P<hometown>[a-zA-Z]*). Nice to meet you!"'
prompt = "Here is a way to present myself, if my name was John and I born in Boston: "

llm = LlamaCPP()
regex_parser = lmformatenforcer.RegexParser(regex)
lm_format_enforcer_fn = build_lm_format_enforcer_function(llm, regex_parser)
with activate_lm_format_enforcer(llm, lm_format_enforcer_fn):
    output = llm.complete(prompt)
```

This will cause the LLM to generate output in the regular expression format that we specified. We can also parse the output to get the named groups:

```python
print(output)
# "Hello, my name is John. I was born in Boston, Nice to meet you!"
print(re.match(regex, output.text).groupdict())
# {'name': 'John', 'hometown': 'Boston'}
```

See [this notebook](../../examples/output_parsing/lmformatenforcer_regular_expressions.ipynb) for more details.
