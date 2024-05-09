import asyncio
from typing import Any, Dict, List, Literal, Optional, TypeAlias

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import create_model, validator
from llama_index.core.graph_stores.types import EntityNode, Relation, Triplet
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.llms.llm import LLM


DEFAULT_ENTITIES = Literal[
    "PRODUCT",
    "MARKET",
    "TECHNOLOGY",
    "EVENT",
    "CONCEPT",
    "ORGANIZATION",
    "PERSON",
    "LOCATION",
    "TIME",
    "MISCELLANEOUS",
]

DEFAULT_RELATIONS = Literal[
    "USED_BY",
    "USED_FOR",
    "LOCATED_IN",
    "PART_OF",
    "WORKED_ON",
    "HAS",
    "IS_A",
    "BORN_IN",
    "DIED_IN",
]

# Which entities can be connected to which relations
DEFAULT_VALIDATION_SCHEMA: Dict[str, Any] = {
    "PRODUCT": (
        "USED_BY",
        "USED_FOR",
        "LOCATED_IN",
        "PART_OF",
        "WORKED_ON",
        "HAS",
        "IS_A",
    ),
    "MARKET": ("LOCATED_IN", "PART_OF", "WORKED_ON", "HAS", "IS_A"),
    "TECHNOLOGY": (
        "USED_BY",
        "USED_FOR",
        "LOCATED_IN",
        "PART_OF",
        "WORKED_ON",
        "HAS",
        "IS_A",
    ),
    "EVENT": ("LOCATED_IN", "PART_OF", "WORKED_ON", "HAS", "IS_A"),
    "CONCEPT": ("USE_BY", "USED_FOR", "PART_OF", "WORKED_ON", "HAS", "IS_A"),
    "ORGANIZATION": ("LOCATED_IN", "PART_OF", "HAS", "IS_A"),
    "PERSON": (
        "BORN_IN",
        "DIED_IN",
        "LOCATED_IN",
        "PART_OF",
        "WORKED_ON",
        "HAS",
        "IS_A",
    ),
    "LOCATION": (
        "LOCATED_IN",
        "PART_OF",
        "HAS",
        "IS_A",
        "DIED_IN",
        "BORN_IN",
        "USED_BY",
        "USED_FOR",
    ),
    "TIME": ("BORN_IN", "DIED_IN", "LOCATED_IN", "PART_OF", "HAS", "IS_A"),
    "MISCELLANEOUS": (
        "USED_BY",
        "USED_FOR",
        "LOCATED_IN",
        "PART_OF",
        "WORKED_ON",
        "HAS",
        "IS_A",
        "BORN_IN",
        "DIED_IN",
    ),
}

DEFAULT_SCHEMA_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    "Give the following text, extract the knowledge graph according to the provided schema. "
    "Try to limit to the output {max_triplets_per_chunk} extracted triplets.s\n"
    "-------\n"
    "{text}\n"
    "-------\n"
)


class SchemaLLMTripletExtractor(TransformComponent):
    """Extract triplets from a graph using a schema."""

    llm: LLM
    extract_prompt: PromptTemplate
    kg_schema_cls: Any
    kg_validation_schema: Dict[str, Any]
    num_workers: int
    max_triplets_per_chunk: int
    show_progress: bool

    def __init__(
        self,
        llm: LLM,
        extract_prompt: PromptTemplate = None,
        possible_entities: Optional[TypeAlias] = None,
        possible_relations: Optional[TypeAlias] = None,
        kg_schema_cls: Any = None,
        kg_validation_schema: Dict[str, str] = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
        show_progress: bool = False,
    ) -> None:
        """Init params."""
        # Build a pydantic model on the fly
        if kg_schema_cls is None:
            possible_entities = possible_entities or DEFAULT_ENTITIES
            entity_cls = create_model(
                "Entity",
                __doc__=(
                    "Entity in a knowledge graph. Only extract entities with types that are listed as valid: "
                    + str(possible_entities)
                ),
                type=(possible_entities, ...),
                value=(str, ...),
            )

            possible_relations = possible_relations or DEFAULT_RELATIONS
            relation_cls = create_model(
                "Relation",
                __doc__=(
                    "Relation in a knowledge graph. Only extract relations with types that are listed as valid: "
                    + str(possible_relations)
                ),
                type=(possible_relations, ...),
            )

            triplet_cls = create_model(
                "Triplet",
                __doc__="Triplet in a knowledge graph.",
                subject=(entity_cls, ...),
                relation=(relation_cls, ...),
                object=(entity_cls, ...),
            )

            def validate(v: Any, values: Any) -> Any:
                """Validate triplets."""
                passing_triplets = []
                for i, triplet in enumerate(v):
                    # cleanup
                    try:
                        for key in triplet:
                            triplet[key]["type"] = triplet[key]["type"].replace(
                                " ", "_"
                            )
                            triplet[key]["type"] = triplet[key]["type"].upper()

                        # validate, skip if invalid
                        _ = triplet_cls(**triplet)
                        passing_triplets.append(v[i])
                    except (KeyError, ValueError):
                        continue

                return passing_triplets

            root = validator("triplets", pre=True)(validate)
            kg_schema_cls = create_model(
                "KGSchema",
                __doc__="Knowledge Graph Schema.",
                __validators__={"validator1": root},
                triplets=(List[triplet_cls], ...),
            )

        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt or DEFAULT_SCHEMA_TRIPLET_EXTRACT_PROMPT,
            kg_schema_cls=kg_schema_cls,
            kg_validation_schema=kg_validation_schema or DEFAULT_VALIDATION_SCHEMA,
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            show_progress=show_progress,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ExtractTripletsFromText"

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Extract triplets from nodes."""
        return asyncio.run(self.acall(nodes, **kwargs))

    def _prune_invalid_triplets(self, kg_schema: Any) -> List[Triplet]:
        """Prune invalid triplets."""
        assert isinstance(kg_schema, self.kg_schema_cls)

        valid_triplets = []
        for triplet in kg_schema.triplets:
            subject = triplet.subject.value
            subject_type = triplet.subject.type

            relation = triplet.relation.type

            obj = triplet.object.value
            obj_type = triplet.object.type

            # check relations
            if relation not in self.kg_validation_schema.get(subject_type, [relation]):
                continue
            if relation not in self.kg_validation_schema.get(obj_type, [relation]):
                continue

            # remove self-references
            if subject.lower() == obj.lower():
                continue

            subj_node = EntityNode(label=subject_type, name=subject)
            obj_node = EntityNode(label=obj_type, name=obj)
            rel_node = Relation(
                label=relation, source_id=subj_node.id, target_id=obj_node.id
            )
            valid_triplets.append((subj_node, rel_node, obj_node))

        return valid_triplets

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triplets from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            kg_schema = await self.llm.astructured_predict(
                self.kg_schema_cls,
                self.extract_prompt,
                text=text,
                max_triplets_per_chunk=self.max_triplets_per_chunk,
            )
            triplets = self._prune_invalid_triplets(kg_schema)
        except ValueError:
            triplets = []

        existing_nodes = node.metadata.pop("nodes", [])
        existing_relations = node.metadata.pop("relations", [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triplets:
            subj.properties = metadata
            obj.properties = metadata
            rel.properties = metadata

            existing_relations.append(rel)
            existing_nodes.append(subj)
            existing_nodes.append(obj)

        node.metadata["nodes"] = existing_nodes
        node.metadata["relations"] = existing_relations

        return node

    async def acall(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Extract triplets from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs, workers=self.num_workers, show_progress=self.show_progress
        )
