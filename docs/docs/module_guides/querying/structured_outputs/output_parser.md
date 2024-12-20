# Output Parsing Modules

LlamaIndex supports integrations with output parsing modules offered
by other frameworks. These output parsing modules can be used in the following ways:

- To provide formatting instructions for any prompt / query (through `output_parser.format`)
- To provide "parsing" for LLM outputs (through `output_parser.parse`)

### Guardrails

Guardrails is an open-source Python package for specification/validation/correction of output schemas. See below for a code example.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.output_parsers.guardrails import GuardrailsOutputParser
from llama_index.llms.openai import OpenAI


# load documents, build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex(documents, chunk_size=512)

# define query / output spec
rail_spec = """
<rail version="0.1">

<output>
    <list name="points" description="Bullet points regarding events in the author's life.">
        <object>
            <string name="explanation" format="one-line" on-fail-one-line="noop" />
            <string name="explanation2" format="one-line" on-fail-one-line="noop" />
            <string name="explanation3" format="one-line" on-fail-one-line="noop" />
        </object>
    </list>
</output>

<prompt>

Query string here.

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none
</prompt>
</rail>
"""

# define output parser
output_parser = GuardrailsOutputParser.from_rail_string(
    rail_spec, llm=OpenAI()
)

# Attach output parser to LLM
llm = OpenAI(output_parser=output_parser)

# obtain a structured response
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(
    "What are the three items the author did growing up?",
)
print(response)
```

Output:

```
{'points': [{'explanation': 'Writing short stories', 'explanation2': 'Programming on an IBM 1401', 'explanation3': 'Using microcomputers'}]}
```

### Langchain

Langchain also offers output parsing modules that you can use within LlamaIndex.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.llms.openai import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# load documents, build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex.from_documents(documents)

# define output schema
response_schemas = [
    ResponseSchema(
        name="Education",
        description="Describes the author's educational experience/background.",
    ),
    ResponseSchema(
        name="Work",
        description="Describes the author's work experience/background.",
    ),
]

# define output parser
lc_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)
output_parser = LangchainOutputParser(lc_output_parser)

# Attach output parser to LLM
llm = OpenAI(output_parser=output_parser)

# obtain a structured response
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(
    "What are a few things the author did growing up?",
)
print(str(response))
```

Output:

```
{'Education': 'Before college, the author wrote short stories and experimented with programming on an IBM 1401.', 'Work': 'The author worked on writing and programming outside of school.'}
```

### Guides

More examples:

- [Guardrails](../../../examples/output_parsing/GuardrailsDemo.ipynb)
- [Langchain](../../../examples/output_parsing/LangchainOutputParserDemo.ipynb)
- [Guidance Pydantic Program](../../../examples/output_parsing/guidance_pydantic_program.ipynb)
- [Guidance Sub-Question](../../../examples/output_parsing/guidance_sub_question.ipynb)
- [Openai Pydantic Program](../../../examples/output_parsing/openai_pydantic_program.ipynb)
