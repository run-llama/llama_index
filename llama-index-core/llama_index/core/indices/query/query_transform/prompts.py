"""Query transform prompts."""


from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

# deprecated, kept for backwards compatibility
"""Decompose prompt for query transformation.

PromptTemplate to "decompose" a query into another query
given the existing context.

Required template variables: `context_str`, `query_str`
"""
DecomposeQueryTransformPrompt = PromptTemplate

"""Step Decompose prompt for query transformation.

PromptTemplate to "decompose" a query into another query
given the existing context + previous reasoning (the previous steps).

Required template variables: `context_str`, `query_str`, `prev_reasoning`
"""
StepDecomposeQueryTransformPrompt = PromptTemplate

"""Image output prompt for query transformation.

PromptTemplate to add instructions for formatting image output.

Required template variables: `query_str`, `image_width`
"""
ImageOutputQueryTransformPrompt = PromptTemplate


DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below. \n"
    "Given the context, return a new question that can be answered from "
    "the context. The question can be the same as the original question, "
    "or a new question that represents a subcomponent of the overall question.\n"
    "As an example: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: What is the current population of the city in which Paul Graham found "
    "his first company, Viaweb?\n"
    "Knowledge source context: Provides information about Paul Graham's "
    "professional career, including the startups he's founded. "
    "New question: In which city did Paul Graham found his first company, Viaweb? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "New question: "
)

DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT = PromptTemplate(
    DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL, prompt_type=PromptType.DECOMPOSE
)


DEFAULT_IMAGE_OUTPUT_TMPL = (
    "{query_str}"
    "Show any image with a HTML <img/> tag with {image_width}."
    'e.g., <image src="data/img.jpg" width="{image_width}" />.'
)

DEFAULT_IMAGE_OUTPUT_PROMPT = PromptTemplate(DEFAULT_IMAGE_OUTPUT_TMPL)


DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below, as "
    "well as previous reasoning steps.\n"
    "Given the context and previous reasoning, return a question that can "
    "be answered from "
    "the context. This question can be the same as the original question, "
    "or this question can represent a subcomponent of the overall question."
    "It should not be irrelevant to the original question.\n"
    "If we cannot extract more information from the context, provide 'None' "
    "as the answer. "
    "Some examples are given below: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides names of the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None\n"
    "Next question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: Who was the winner of the 2020 Australian Open?\n"
    "Knowledge source context: Provides names of the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None.\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: None"
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open - includes biographical information for each winner\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: How many Grand Slam titles does Novak Djokovic have? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "Previous reasoning: {prev_reasoning}\n"
    "New question: "
)

DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT = PromptTemplate(
    DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_TMPL
)
