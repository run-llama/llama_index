"""Prompts from evaporate repo.


Full credits go to: https://github.com/HazyResearch/evaporate


"""

from gpt_index.prompts.prompts import Prompt
from gpt_index.prompts.prompt_type import PromptType
from typing import List


class SchemaIDPrompt(Prompt):
    """Pandas prompt. Convert query to python code.

    Required template variables: `chunk`, `topic`.

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    prompt_type: PromptType = PromptType.CUSTOM
    input_variables: List[str] = ["chunk", "topic"]


class FnGeneratePrompt(Prompt):
    """Function generation prompt. Generate a function from existing text.

    Required template variables: `context_str`, `query_str`,
        `attribute`, `function_field`.

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    prompt_type: PromptType = PromptType.CUSTOM
    input_variables: List[str] = [
        "context_str",
        "query_str",
        "attribute",
        "function_field",
    ]


# used for schema identification
SCHEMA_ID_PROMPT_TMPL = f"""Sample text:
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
{{chunk:}}

Question: List all relevant attributes about '{{topic:}}' that are exactly mentioned in this sample text if any. 
Answer:"""

SCHEMA_ID_PROMPT = SchemaIDPrompt(SCHEMA_ID_PROMPT_TMPL)


# used for function generation

FN_GENERATION_PROMPT_TMPL = f"""Here is a sample of text:

{{context_str:}}


Question: {{query_str:}}

Return the result as a list.

import re

def get_{{function_field:}}_field(text: str):
    \"""
    Function to extract the "{{attribute:}} field".
    \"""
    """

FN_GENERATION_PROMPT = FnGeneratePrompt(FN_GENERATION_PROMPT_TMPL)

# METADATA_GENERATION_PROMPT_TMPL = f"""Here is a sample of text:

# {{chunk:}}


# Question: Write a python function to extract the entire "{{attribute:}}" field from text, but not any other metadata. Return the result as a list.


# import re

# def get_{{function_field:}}_field(text: str):
#     \"""
#     Function to extract the "{{attribute:}} field".
#     \"""
# """

# METADATA_GENERATION_PROMPT_TMPL2 = f"""Here is a file sample:

# DESCRIPTION: This file answers the question, "How do I sort a dictionary by value?"
# DATES MODIFIED: The file was modified on the following dates:
# 2009-03-05T00:49:05
# 2019-04-07T00:22:14
# 2011-11-20T04:21:49
# USERS: The users who modified the file are:
# Jeff Jacobs
# Richard Smith
# Julia D'Angelo
# Rebecca Matthews
# FILE TYPE: This is a text file.

# Question: Write a python function called "get_dates_modified_field" to extract the "DATES MODIFIED" field from the text. Include any imports.

# import re

# def get_dates_modified_field(text: str):
#     \"""
#     Function to extract the dates modified.
#     \"""
#     parts= text.split("USERS")[0].split("DATES MODIFIED")[-1]
#     pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
#     return re.findall(pattern, text)

# ----

# Here is a file sample:

# <title>U.S. GDP Rose 2.9% in the Fourth Quarter After a Year of High Inflation - WSJ</title>
# <meta property="og:url" content="https://www.wsj.com/articles/us-gdp-economic-growth-fourth-quarter-2022-11674683034"/>
# <meta name="article.published" content="2023-01-26T10:30:00Z"/><meta itemProp="datePublished" content="2023-01-26T10:30:00Z"/>
# <meta name="article.created" content="2023-01-26T10:30:00Z"/><meta itemProp="dateCreated" content="2023-01-26T10:30:00Z"/>
# <meta name="dateLastPubbed" content="2023-01-31T19:17:00Z"/><meta name="author" content="Sarah Chaney Cambon"/>

# Question: Write a python function called "get_date_published_field" to extract the "datePublished" field from the text. Include any imports.

# from bs4 import BeautifulSoup

# def get_date_published_field(text: str):
#     \"""
#     Function to extract the date published.
#     \"""
#     soup = BeautifulSoup(text, parser="html.parser")
#     date_published_field = soup.find('meta', itemprop="datePublished")
#     date_published_field = date_published_field['content']
#     return date_published_field

# ----

# Here is a sample of text:

# {{chunk:}}

# Question: Write a python function called "get_{{function_field:}}_field" to extract the "{{attribute:}}" field from the text. Include any imports."""
# ]
