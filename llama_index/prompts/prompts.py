"""Subclasses from base prompt."""
from typing import List

from llama_index.prompts.base import Prompt

# deprecated, kept for backward compatibility
SummaryPrompt = Prompt
TreeInsertPrompt = Prompt
TreeSelectPrompt = Prompt
TreeSelectMultiplePrompt = Prompt
RefinePrompt = Prompt
QuestionAnswerPrompt = Prompt
KeywordExtractPrompt = Prompt
QueryKeywordExtractPrompt = Prompt
SchemaExtractPrompt = Prompt
TextToSQLPrompt = Prompt
TableContextPrompt = Prompt
RefineTableContextPrompt = Prompt
KnowledgeGraphPrompt = Prompt 
SimpleInputPrompt = Prompt
PandasPrompt = Prompt