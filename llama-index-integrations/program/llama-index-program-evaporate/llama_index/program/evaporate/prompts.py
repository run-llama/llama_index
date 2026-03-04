"""
Prompts from evaporate repo.


Full credits go to: https://github.com/HazyResearch/evaporate


"""

from llama_index.core.prompts import PromptTemplate

# deprecated, kept for backward compatibility

"""Pandas PromptTemplate. Convert query to python code.

Required template variables: `chunk`, `topic`.

Args:
    template (str): Template for the PromptTemplate.
    **prompt_kwargs: Keyword arguments for the PromptTemplate.

"""
SchemaIDPrompt = PromptTemplate

"""Function generation PromptTemplate. Generate a function from existing text.

Required template variables: `context_str`, `query_str`,
    `attribute`, `function_field`.

Args:
    template (str): Template for the PromptTemplate.
    **prompt_kwargs: Keyword arguments for the PromptTemplate.

"""
FnGeneratePrompt = PromptTemplate

# used for schema identification
SCHEMA_ID_PROMPT_TMPL = """Sample text:
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="text-indent:-0.9em;margin-left:1.2em;font-weight:normal;">•&nbsp;<a href="/wiki/Monarchy_of_Canada" title="Monarchy of Canada">Monarch</a> </div></th><td class="infobox-data"><a href="/wiki/Charles_III" title="Charles III">Charles III</a></td></tr>
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="text-indent:-0.9em;margin-left:1.2em;font-weight:normal;">•&nbsp;<span class="nowrap"><a href="/wiki/Governor_General_of_Canada" title="Governor General of Canada">Governor General</a></span> </div></th><td class="infobox-data"><a href="/wiki/Mary_Simon" title="Mary Simon">Mary Simon</a></td></tr>
<b>Provinces and Territories</b class='navlinking countries'>
<ul>
<li>Saskatchewan</li>
<li>Manitoba</li>
<li>Ontario</li>
<li>Quebec</li>
<li>New Brunswick</li>
<li>Prince Edward Island</li>
<li>Nova Scotia</li>
<li>Newfoundland and Labrador</li>
<li>Yukon</li>
<li>Nunavut</li>
<li>Northwest Territories</li>
</ul>

Question: List all relevant attributes about 'Canada' that are exactly mentioned in this sample text if any.
Answer:
- Monarch: Charles III
- Governor General: Mary Simon
- Provinces and Territories: Saskatchewan, Manitoba, Ontario, Quebec, New Brunswick, Prince Edward Island, Nova Scotia, Newfoundland and Labrador, Yukon, Nunavut, Northwest Territories

----

Sample text:
Patient birth date: 1990-01-01
Prescribed medication: aspirin, ibuprofen, acetaminophen
Prescribed dosage: 1 tablet, 2 tablets, 3 tablets
Doctor's name: Dr. Burns
Date of discharge: 2020-01-01
Hospital address: 123 Main Street, New York, NY 10001

Question: List all relevant attributes about 'medications' that are exactly mentioned in this sample text if any.
Answer:
- Prescribed medication: aspirin, ibuprofen, acetaminophen
- Prescribed dosage: 1 tablet, 2 tablets, 3 tablets

----

Sample text:
{chunk:}

Question: List all relevant attributes about '{topic:}' that are exactly mentioned in this sample text if any.
Answer:"""

SCHEMA_ID_PROMPT = PromptTemplate(SCHEMA_ID_PROMPT_TMPL)


# used for function generation

FN_GENERATION_PROMPT_TMPL = """Here is a sample of text:

{context_str:}


Question: {query_str:}

Given the function signature, write Python code to extract the
"{attribute:}" field from the text.
Return the result as a single value (string, int, float), and not a list.
Make sure there is a return statement in the code. Do not leave out a return statement.
{expected_output_str:}

import re

def get_{function_field:}_field(text: str):
    \"""
    Function to extract the "{attribute:} field", and return the result
    as a single value.
    \"""
    """

FN_GENERATION_PROMPT = PromptTemplate(FN_GENERATION_PROMPT_TMPL)


FN_GENERATION_LIST_PROMPT_TMPL = """Here is a sample of text:

{context_str:}


Question: {query_str:}

Given the function signature, write Python code to extract the
"{attribute:}" field from the text.
Return the result as a list of values (if there is just one item, return a single \
element list).
Make sure there is a return statement in the code. Do not leave out a return statement.
{expected_output_str:}

import re

def get_{function_field:}_field(text: str) -> List:
    \"""
    Function to extract the "{attribute:} field", and return the result
    as a single value.
    \"""
    """

FN_GENERATION_LIST_PROMPT = PromptTemplate(FN_GENERATION_LIST_PROMPT_TMPL)

DEFAULT_EXPECTED_OUTPUT_PREFIX_TMPL = (
    "Here is the expected output on the text after running the function. "
    "Please do not write a function that would return a different output. "
    "Expected output: "
)


DEFAULT_FIELD_EXTRACT_QUERY_TMPL = (
    'Write a python function to extract the entire "{field}" field from text, '
    "but not any other metadata. Return the result as a list."
)
