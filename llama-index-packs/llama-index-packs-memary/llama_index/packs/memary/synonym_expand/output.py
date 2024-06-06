from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class SynonymOutput(BaseModel):
    synonyms: List[str] = Field(
        description=
        "Synonyms of keywords provided, make every letter lowercase except for the first letter"
    )
