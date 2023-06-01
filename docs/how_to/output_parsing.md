# 🔢 Output Parsing

LLM output/validation capabilities are crucial to LlamaIndex in the following areas:
- **Document retrieval**: Many data structures within LlamaIndex rely on LLM calls with a specific schema for Document retrieval. For instance, the tree index expects LLM calls to be in the format "ANSWER: (number)".
- **Response synthesis**: Users may expect that the final response contains some degree of structure (e.g. a JSON output, a formatted SQL query, etc.)

LlamaIndex supports integrations with output parsing modules offered
by other frameworks. These output parsing modules can be used in the following ways:
- To provide formatting instructions for any prompt / query (through `output_parser.format`)
- To provide "parsing" for LLM outputs (through `output_parser.parse`)


### Guardrails

Guardrails is an open-source Python package for specification/validation/correction of output schemas. See below for a code example.


```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.output_parsers import GuardrailsOutputParser
from llama_index.llm_predictor import StructuredLLMPredictor
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL, DEFAULT_REFINE_PROMPT_TMPL


# load documents, build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex(documents, chunk_size=512)
llm_predictor = StructuredLLMPredictor()


# specify StructuredLLMPredictor
# this is a special LLMPredictor that allows for structured outputs

# define query / output spec
rail_spec = ("""
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
""")

# define output parser
output_parser = GuardrailsOutputParser.from_rail_string(rail_spec, llm=llm_predictor.llm)

# format each prompt with output parser instructions
fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)

qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

# obtain a structured response
query_engine = index.as_query_engine(
    service_context=ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    ),
    text_qa_temjlate=qa_prompt, 
    refine_template=refine_prompt, 
)
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
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.output_parsers import LangchainOutputParser
from llama_index.llm_predictor import StructuredLLMPredictor
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL, DEFAULT_REFINE_PROMPT_TMPL
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# load documents, build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
llm_predictor = StructuredLLMPredictor()

# define output schema
response_schemas = [
    ResponseSchema(name="Education", description="Describes the author's educational experience/background."),
    ResponseSchema(name="Work", description="Describes the author's work experience/background.")
]

# define output parser
lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)

# format each prompt with output parser instructions
fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

# query index
query_engine = index.as_query_engine(
    service_context=ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    ),
    text_qa_template=qa_prompt, 
    refine_template=refine_prompt, 
)
response = query_engine.query(
    "What are a few things the author did growing up?", 
)
print(str(response))
```

Output:

```
{'Education': 'Before college, the author wrote short stories and experimented with programming on an IBM 1401.', 'Work': 'The author worked on writing and programming outside of school.'}
```


```{toctree}
---
caption: Examples
maxdepth: 1
---

../examples/output_parsing/GuardrailsDemo.ipynb
../examples/output_parsing/LangchainOutputParserDemo.ipynb
```
