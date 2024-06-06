from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List
import os
from llama_index.packs.memary.synonym_expand.output import SynonymOutput
from dotenv import load_dotenv


def custom_synonym_expand_fn(keywords: str) -> List[str]:
    load_dotenv()
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_KEY"), temperature=0)
    parser = JsonOutputParser(pydantic_object=SynonymOutput)

    template = """
    You are an expert synonym exapnding system. Find synonyms or words commonly used in place to reference the same word for every word in the list:

    Some examples are:
    - a synonym for Palantir may be Palantir technologies or Palantir technologies inc.
    - a synonym for Austin may be Austin texas
    - a synonym for Taylor swift may be Taylor 
    - a synonym for Winter park may be Winter park resort
    
    Format: {format_instructions}

    Text: {keywords}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["keywords"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    chain = prompt | llm | parser
    result = chain.invoke({"keywords": keywords})

    l = []
    for category in result:
        for synonym in result[category]:
            l.append(synonym.capitalize())

    return l


# testing
# print(custom_synonym_expand_fn("[Nvidia]"))
