import re
from typing import Any, List, Optional, Union

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.llms import LLM
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.prompts import (
    BasePromptTemplate,
    RichPromptTemplate,
    PromptTemplate,
)
from llama_index.core.settings import Settings

DEFAULT_FACT_EXTRACT_PROMPT = RichPromptTemplate("""You are a precise fact extraction system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the conversation segment provided prior to this message
2. Extract specific, concrete facts the user has disclosed or important information discovered
3. Focus on factual information like preferences, personal details, requirements, constraints, or context
4. Format each fact as a separate <fact> XML tag
5. Do not include opinions, summaries, or interpretations - only extract explicit information
6. Do not duplicate facts that are already in the existing facts list

<existing_facts>
{{ existing_facts }}
</existing_facts>

Return ONLY the extracted facts in this exact format:
<facts>
  <fact>Specific fact 1</fact>
  <fact>Specific fact 2</fact>
  <!-- More facts as needed -->
</facts>

If no new facts are present, return: <facts></facts>""")

DEFAULT_FACT_CONDENSE_PROMPT = RichPromptTemplate("""You are a precise fact condensing system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the current list of existing facts
2. Condense the facts into a more concise list, less than {{ max_facts }} facts
3. Focus on factual information like preferences, personal details, requirements, constraints, or context
4. Format each fact as a separate <fact> XML tag
5. Do not include opinions, summaries, or interpretations - only extract explicit information
6. Do not duplicate facts that are already in the existing facts list

<existing_facts>
{{ existing_facts }}
</existing_facts>

Return ONLY the condensed facts in this exact format:
<facts>
  <fact>Specific fact 1</fact>
  <fact>Specific fact 2</fact>
  <!-- More facts as needed -->
</facts>

If no new facts are present, return: <facts></facts>""")


def get_default_llm() -> LLM:
    return Settings.llm


class FactExtractionMemoryBlock(BaseMemoryBlock[str]):
    """
    A memory block that extracts key facts from conversation history using an LLM.

    This block identifies and stores discrete facts disclosed during the conversation,
    structuring them in XML format for easy parsing and retrieval.
    """

    name: str = Field(
        default="ExtractedFacts", description="The name of the memory block."
    )
    llm: LLM = Field(
        default_factory=get_default_llm,
        description="The LLM to use for fact extraction.",
    )
    facts: List[str] = Field(
        default_factory=list,
        description="List of extracted facts from the conversation.",
    )
    max_facts: int = Field(
        default=50, description="The maximum number of facts to store."
    )
    fact_extraction_prompt_template: BasePromptTemplate = Field(
        default=DEFAULT_FACT_EXTRACT_PROMPT,
        description="Template for the fact extraction prompt.",
    )
    fact_condense_prompt_template: BasePromptTemplate = Field(
        default=DEFAULT_FACT_CONDENSE_PROMPT,
        description="Template for the fact condense prompt.",
    )

    @field_validator("fact_extraction_prompt_template", mode="before")
    @classmethod
    def validate_fact_extraction_prompt_template(
        cls, v: Union[str, BasePromptTemplate]
    ) -> BasePromptTemplate:
        if isinstance(v, str):
            if "{{" in v and "}}" in v:
                v = RichPromptTemplate(v)
            else:
                v = PromptTemplate(v)
        return v

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Return the current facts as formatted text."""
        if not self.facts:
            return ""

        return "\n".join([f"<fact>{fact}</fact>" for fact in self.facts])

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Extract facts from new messages and add them to the facts list."""
        # Skip if no messages
        if not messages:
            return

        # Format existing facts for the prompt
        existing_facts_text = ""
        if self.facts:
            existing_facts_text = "\n".join(
                [f"<fact>{fact}</fact>" for fact in self.facts]
            )

        # Create the prompt
        prompt_messages = self.fact_extraction_prompt_template.format_messages(
            existing_facts=existing_facts_text,
        )

        # Get the facts extraction
        response = await self.llm.achat(messages=[*messages, *prompt_messages])

        # Parse the XML response to extract facts
        facts_text = response.message.content or ""
        new_facts = self._parse_facts_xml(facts_text)

        # Add new facts to the list, avoiding exact-match duplicates
        for fact in new_facts:
            if fact not in self.facts:
                self.facts.append(fact)

        # Condense the facts if they exceed the max_facts
        if len(self.facts) > self.max_facts:
            existing_facts_text = "\n".join(
                [f"<fact>{fact}</fact>" for fact in self.facts]
            )

            prompt_messages = self.fact_condense_prompt_template.format_messages(
                existing_facts=existing_facts_text,
                max_facts=self.max_facts,
            )
            response = await self.llm.achat(messages=[*messages, *prompt_messages])
            new_facts = self._parse_facts_xml(response.message.content or "")
            self.facts = new_facts

    def _parse_facts_xml(self, xml_text: str) -> List[str]:
        """Parse facts from XML format."""
        facts = []

        # Extract content between <fact> tags
        pattern = r"<fact>(.*?)</fact>"
        matches = re.findall(pattern, xml_text, re.DOTALL)

        # Clean up extracted facts
        for match in matches:
            fact = match.strip()
            if fact:
                facts.append(fact)

        return facts
