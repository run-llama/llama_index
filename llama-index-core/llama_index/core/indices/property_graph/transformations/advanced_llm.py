import asyncio
import json
from typing import Any, Callable, List, Optional, Union, Literal

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
    DEFAULT_ADVANCED_EXTRACT_PROMPT,
)



def default_parse_advanced_triplets(llm_output: str) -> List[tuple]:
    """
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.

    Args:
        llm_output (str): The JSON string output from the LLM.

    Returns:
        List[tuple]: A list of (head, relation, tail) triplets, where head and tail are EntityNodes,
                     and relation is a Relation object.
    """
    try:
        parsed_output = json.loads(llm_output)
    except json.JSONDecodeError:
        return []

    triplets = []
    for item in parsed_output:
        head = EntityNode(name=item['head'], label=item['head_type'])
        tail = EntityNode(name=item['tail'], label=item['tail_type'])
        relation = Relation(source_id=head.id, target_id=tail.id, label=item['relation'])
        triplets.append((head, relation, tail))
    
    return triplets

class AdvancedLLMPathExtractor(TransformComponent):
    """
    AdvancedLLMPathExtractor is a component for extracting structured information from text
    to build a knowledge graph. It uses an LLM to identify entities and their relationships,
    with the ability to infer entity types and expand upon an initial ontology.

    This extractor improves upon SimpleLLMPathExtractor  by:
    1.

    This extractor is different from SchemaLLMPathExtractor because : 

    Attributes:
        llm (LLM): The language model used for extraction.
        extract_prompt (PromptTemplate): The prompt template used to guide the LLM.
        parse_fn (Callable): Function to parse the LLM output into triplets.
        num_workers (int): Number of workers for parallel processing.
        max_triplets_per_chunk (int): Maximum number of triplets to extract per text chunk.
        allowed_entity_types (List[str]): List of initial entity types for the ontology.
        allowed_relation_types (List[str]): List of initial relation types for the ontology.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_triplets_per_chunk: int
    allowed_entity_types: List[str]
    allowed_relation_types: List[str]

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_advanced_triplets,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_relation_types: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the AdvancedLLMPathExtractor.

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
            extract_prompt = DEFAULT_ADVANCED_EXTRACT_PROMPT

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            allowed_entity_types=allowed_entity_types or [],
            allowed_relation_types=allowed_relation_types or [],
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "AdvancedLLMPathExtractor"

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
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_triplets_per_chunk,
                allowed_entity_types=", ".join(self.allowed_entity_types) if self.allowed_entity_types else "No initial entity types provided",
                allowed_relation_types=", ".join(self.allowed_relation_types) if self.allowed_relation_types else "No initial relation types provided",
            )
            triplets = self.parse_fn(llm_response)
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            triplets = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triplets:
            subj.properties = metadata
            obj.properties = metadata
            rel.properties = metadata

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