"""Default prompt selectors."""
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.chat_prompts import (
    CHAT_REFINE_PROMPT,
    CHAT_REFINE_TABLE_CONTEXT_PROMPT,
    CHAT_TEXT_QA_PROMPT,
    CHAT_TREE_SUMMARIZE_PROMPT,
)
from llama_index.core.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_TREE_SUMMARIZE_PROMPT,
)
from llama_index.core.prompts.utils import is_chat_model

try:
    from llama_index.llms.cohere import (
        is_cohere_model,
        COHERE_QA_TEMPLATE,
        COHERE_REFINE_TEMPLATE,
        COHERE_TREE_SUMMARIZE_TEMPLATE,
        COHERE_REFINE_TABLE_CONTEXT_PROMPT,
    )  # pants: no-infer-dep
except ImportError:
    COHERE_QA_TEMPLATE = None
    COHERE_REFINE_TEMPLATE = None
    COHERE_TREE_SUMMARIZE_TEMPLATE = None
    COHERE_REFINE_TABLE_CONTEXT_PROMPT = None

# Define prompt selectors for Text QA, Tree Summarize, Refine, and Refine Table.
# Note: Cohere models accept a special argument `documents` for RAG calls. To pass on retrieved documents to the `documents` argument,
# specialised templates have been defined. The conditionals below ensure that these templates are called by default when a retriever
# is called with a Cohere model for generator.

# Text QA
default_text_qa_conditionals = [(is_chat_model, CHAT_TEXT_QA_PROMPT)]
if COHERE_QA_TEMPLATE is not None:
    default_text_qa_conditionals = [
        (is_cohere_model, COHERE_QA_TEMPLATE),
        (is_chat_model, CHAT_TEXT_QA_PROMPT),
    ]

DEFAULT_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TEXT_QA_PROMPT,
    conditionals=default_text_qa_conditionals,
)

# Tree Summarize
default_tree_summarize_conditionals = [(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)]
if COHERE_TREE_SUMMARIZE_TEMPLATE is not None:
    default_tree_summarize_conditionals = [
        (is_cohere_model, COHERE_TREE_SUMMARIZE_TEMPLATE),
        (is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT),
    ]

DEFAULT_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TREE_SUMMARIZE_PROMPT,
    conditionals=default_tree_summarize_conditionals,
)

# Refine
default_refine_conditionals = [(is_chat_model, CHAT_REFINE_PROMPT)]
if COHERE_REFINE_TEMPLATE is not None:
    default_refine_conditionals = [
        (is_cohere_model, COHERE_REFINE_TEMPLATE),
        (is_chat_model, CHAT_REFINE_PROMPT),
    ]

DEFAULT_REFINE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_PROMPT,
    conditionals=default_refine_conditionals,
)

# Refine Table Context
default_refine_table_conditionals = [(is_chat_model, CHAT_REFINE_TABLE_CONTEXT_PROMPT)]
if COHERE_REFINE_TABLE_CONTEXT_PROMPT is not None:
    default_refine_table_conditionals = [
        (is_cohere_model, COHERE_REFINE_TABLE_CONTEXT_PROMPT),
        (is_chat_model, CHAT_REFINE_TABLE_CONTEXT_PROMPT),
    ]

DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
    conditionals=default_refine_table_conditionals,
)
