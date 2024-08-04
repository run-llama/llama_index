import asyncio
from typing import Any, Callable, List, Optional, Union, Tuple
import re
import json
from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)

from llama_index.core.prompts.default_prompts import (
    DEFAULT_DYNAMIC_EXTRACT_PROMPT,
    DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT,
)


def default_parse_dynamic_triplets(
    llm_output: str,
) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.
    This function is flexible and can handle various output formats.

    Args:
        llm_output (str): The output from the LLM, which may be JSON-like or plain text.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of triplets.
    """
    triplets = []

    try:
        # Attempt to parse the output as JSON
        data = json.loads(llm_output)
        for item in data:
            head = item.get("head")
            head_type = item.get("head_type")
            relation = item.get("relation")
            tail = item.get("tail")
            tail_type = item.get("tail_type")

            if head and head_type and relation and tail and tail_type:
                head_node = EntityNode(name=head, label=head_type)
                tail_node = EntityNode(name=tail, label=tail_type)
                relation_node = Relation(
                    source_id=head_node.id, target_id=tail_node.id, label=relation
                )
                triplets.append((head_node, relation_node, tail_node))

    except json.JSONDecodeError:
        # Flexible pattern to match the key-value pairs for head, head_type, relation, tail, and tail_type
        pattern = r'[\{"\']head[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']head_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']relation[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']tail[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']tail_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']'

        # Find all matches in the output
        matches = re.findall(pattern, llm_output)

        for match in matches:
            head, head_type, relation, tail, tail_type = match
            head_node = EntityNode(name=head, label=head_type)
            tail_node = EntityNode(name=tail, label=tail_type)
            relation_node = Relation(
                source_id=head_node.id, target_id=tail_node.id, label=relation
            )
            triplets.append((head_node, relation_node, tail_node))
    return triplets


def default_parse_dynamic_triplets_with_props(
    llm_output: str,
) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.
    This function is flexible and can handle various output formats.

    Args:
        llm_output (str): The output from the LLM, which may be JSON-like or plain text.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of triplets.
    """
    triplets = []

    try:
        # Attempt to parse the output as JSON
        data = json.loads(llm_output)
        for item in data:
            head = item.get("head")
            head_type = item.get("head_type")
            head_props = item.get("head_props", {})
            relation = item.get("relation")
            relation_props = item.get("relation_props", {})
            tail = item.get("tail")
            tail_type = item.get("tail_type")
            tail_props = item.get("tail_props", {})

            if head and head_type and relation and tail and tail_type:
                head_node = EntityNode(
                    name=head, label=head_type, properties=head_props
                )
                tail_node = EntityNode(
                    name=tail, label=tail_type, properties=tail_props
                )
                relation_node = Relation(
                    source_id=head_node.id,
                    target_id=tail_node.id,
                    label=relation,
                    properties=relation_props,
                )
                triplets.append((head_node, relation_node, tail_node))
    except json.JSONDecodeError:
        # Flexible pattern to match the key-value pairs for head, head_type, head_props, relation, relation_props, tail, tail_type, and tail_props
        pattern = r'[\{"\']head[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']head_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']head_props[\}"\']\s*:\s*\{(.*?)\}\s*,\s*[\{"\']relation[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']relation_props[\}"\']\s*:\s*\{(.*?)\}\s*,\s*[\{"\']tail[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']tail_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']tail_props[\}"\']\s*:\s*\{(.*?)\}\s*'

        # Find all matches in the output
        matches = re.findall(pattern, llm_output)

        for match in matches:
            (
                head,
                head_type,
                head_props,
                relation,
                relation_props,
                tail,
                tail_type,
                tail_props,
            ) = match

            # Use more robust parsing for properties
            def parse_props(props_str):
                try:
                    # Handle mixed quotes and convert to a proper dictionary
                    props_str = props_str.replace("'", '"')
                    return json.loads(f"{{{props_str}}}")
                except json.JSONDecodeError:
                    return {}

            head_props_dict = parse_props(head_props)
            relation_props_dict = parse_props(relation_props)
            tail_props_dict = parse_props(tail_props)

            head_node = EntityNode(
                name=head, label=head_type, properties=head_props_dict
            )
            tail_node = EntityNode(
                name=tail, label=tail_type, properties=tail_props_dict
            )
            relation_node = Relation(
                source_id=head_node.id,
                target_id=tail_node.id,
                label=relation,
                properties=relation_props_dict,
            )
            triplets.append((head_node, relation_node, tail_node))
    return triplets


class DynamicLLMPathExtractor(TransformComponent):
    """
    DynamicLLMPathExtractor is a component for extracting structured information from text
    to build a knowledge graph. It uses an LLM to identify entities and their relationships,
    with the ability to infer entity types and expand upon an initial ontology.

    This extractor improves upon SimpleLLMPathExtractor by:
    1. Detecting entity types instead of labeling them generically as "entity" and "chunk".
    2. Accepting an initial ontology as input, specifying desired nodes and relationships.
    3. Encouraging ontology expansion through its prompt design.

    This extractor differs from SchemaLLMPathExtractor because:
    1. It interprets the passed possible entities and relations as an initial ontology.
    2. It encourages expansion of the initial ontology in the prompt.
    3. It aims for flexibility in knowledge graph construction while still providing guidance.

    Attributes:
        llm (LLM): The language model used for extraction.
        extract_prompt (PromptTemplate): The prompt template used to guide the LLM.
        parse_fn (Callable): Function to parse the LLM output into triplets.
        num_workers (int): Number of workers for parallel processing.
        max_triplets_per_chunk (int): Maximum number of triplets to extract per text chunk.
        allowed_entity_types (List[str]): List of initial entity types for the ontology.
        allowed_entity_props (Optional[Union[List[str], List[Tuple[str, str]]]]):
            List of initial entity properties for the ontology.
            Can be either property names or tuples of (name, description).
        allowed_relation_types (List[str]): List of initial relation types for the ontology.
        allowed_relation_props (Optional[Union[List[str], List[Tuple[str, str]]]]):
            List of initial relation properties for the ontology.
            Can be either property names or tuples of (name, description).
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_triplets_per_chunk: int
    allowed_entity_types: List[str]
    allowed_entity_props: List[str]
    allowed_relation_types: Optional[List[str]]
    allowed_relation_props: Optional[List[str]]

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Optional[Callable] = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_entity_props: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
        allowed_relation_types: Optional[List[str]] = None,
        allowed_relation_props: Optional[
            Union[List[str], List[Tuple[str, str]]]
        ] = None,
    ) -> None:
        """
        Initialize the DynamicLLMPathExtractor.

        Args:
            llm (Optional[LLM]): The language model to use. If None, uses the default from Settings.
            extract_prompt (Optional[Union[str, PromptTemplate]]): The prompt template to use.
            parse_fn (Callable): Function to parse LLM output into triplets.
            max_triplets_per_chunk (int): Maximum number of triplets to extract per chunk.
            num_workers (int): Number of workers for parallel processing.
            allowed_entity_types (Optional[List[str]]): List of initial entity types for the ontology.
            allowed_relation_types (Optional[List[str]]): List of initial relation types for the ontology.
        """
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        if extract_prompt is None:
            if allowed_entity_props is not None or allowed_relation_props is not None:
                extract_prompt = DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT
            else:
                extract_prompt = DEFAULT_DYNAMIC_EXTRACT_PROMPT

        if parse_fn is None:
            if allowed_entity_props is not None or allowed_relation_props is not None:
                parse_fn = default_parse_dynamic_triplets_with_props
            else:
                parse_fn = default_parse_dynamic_triplets

        # convert props to name -> description format if needed
        if allowed_entity_props and isinstance(allowed_entity_props[0], tuple):
            allowed_entity_props = [
                f"Property `{k}` with description ({v})"
                for k, v in allowed_entity_props
            ]

        if allowed_relation_props and isinstance(allowed_relation_props[0], tuple):
            allowed_relation_props = [
                f"Property `{k}` with description ({v})"
                for k, v in allowed_relation_props
            ]

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            allowed_entity_types=allowed_entity_types or [],
            allowed_entity_props=allowed_entity_props or [],
            allowed_relation_types=allowed_relation_types or [],
            allowed_relation_props=allowed_relation_props or [],
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "DynamicLLMPathExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Extract triples from nodes.

        Args:
            nodes (List[BaseNode]): List of nodes to process.
            show_progress (bool): Whether to show a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            List[BaseNode]: Processed nodes with extracted information.
        """
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _apredict_without_props(self, text: str) -> str:
        """
        Asynchronously predict triples from text without properties.

        Args:
            text (str): The text to process.

        Returns:
            str: The predicted triples.
        """
        return await self.llm.apredict(
            self.extract_prompt,
            text=text,
            max_knowledge_triplets=self.max_triplets_per_chunk,
            allowed_entity_types=", ".join(self.allowed_entity_types)
            if len(self.allowed_entity_types) > 0
            else "No entity types provided, You are free to define them.",
            allowed_relation_types=", ".join(self.allowed_relation_types)
            if len(self.allowed_relation_types) > 0
            else "No relation types provided, You are free to define them.",
        )

    async def _apredict_with_props(self, text: str) -> str:
        """
        Asynchronously predict triples from text with properties.

        Args:
            text (str): The text to process.

        Returns:
            str: The predicted triples.
        """
        return await self.llm.apredict(
            self.extract_prompt,
            text=text,
            max_knowledge_triplets=self.max_triplets_per_chunk,
            allowed_entity_types=", ".join(self.allowed_entity_types)
            if len(self.allowed_entity_types) > 0
            else "No entity types provided, You are free to define them.",
            allowed_relation_types=", ".join(self.allowed_relation_types)
            if len(self.allowed_relation_types) > 0
            else "No relation types provided, You are free to define them.",
            allowed_entity_properties=", ".join(self.allowed_entity_props)
            if self.allowed_entity_props
            else "No entity properties provided, You are free to define them.",
            allowed_relation_properties=", ".join(self.allowed_relation_props)
            if self.allowed_relation_props
            else "No relation properties provided, You are free to define them.",
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """
        Asynchronously extract triples from a single node.

        Args:
            node (BaseNode): The node to process.

        Returns:
            BaseNode: The processed node with extracted information.
        """
        text = node.get_content(metadata_mode="llm")
        try:
            if (
                self.allowed_entity_props is not None
                and self.allowed_relation_props is not None
            ):
                llm_response = await self._apredict_with_props(text)
            else:
                llm_response = await self._apredict_without_props(text)

            triplets = self.parse_fn(llm_response)
        except Exception as e:
            print(f"Error during extraction: {e!s}")
            triplets = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triplets:
            subj.properties.update(metadata)
            obj.properties.update(metadata)
            rel.properties.update(metadata)

            existing_nodes.extend([subj, obj])
            existing_relations.append(rel)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Asynchronously extract triples from multiple nodes.

        Args:
            nodes (List[BaseNode]): List of nodes to process.
            show_progress (bool): Whether to show a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            List[BaseNode]: Processed nodes with extracted information.
        """
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting and inferring knowledge graph from text",
        )
