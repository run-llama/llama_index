import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

try:
    from typing import TypeAlias
except ImportError:
    # python 3.8 and 3.9 compatibility
    TypeAlias = Any

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import create_model, validator, Field
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    Triplet,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
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
    "HAS_ALIAS",
]

# Convert the above dict schema into a list of triples
Triple = Tuple[str, str, str]
DEFAULT_VALIDATION_SCHEMA: List[Triple] = [
    ("PRODUCT", "USED_BY", "PRODUCT"),
    ("PRODUCT", "USED_FOR", "MARKET"),
    ("PRODUCT", "HAS", "TECHNOLOGY"),
    ("MARKET", "LOCATED_IN", "LOCATION"),
    ("MARKET", "HAS", "TECHNOLOGY"),
    ("TECHNOLOGY", "USED_BY", "PRODUCT"),
    ("TECHNOLOGY", "USED_FOR", "MARKET"),
    ("TECHNOLOGY", "LOCATED_IN", "LOCATION"),
    ("TECHNOLOGY", "PART_OF", "ORGANIZATION"),
    ("TECHNOLOGY", "IS_A", "PRODUCT"),
    ("EVENT", "LOCATED_IN", "LOCATION"),
    ("EVENT", "PART_OF", "ORGANIZATION"),
    ("CONCEPT", "USED_BY", "TECHNOLOGY"),
    ("CONCEPT", "USED_FOR", "PRODUCT"),
    ("ORGANIZATION", "LOCATED_IN", "LOCATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "MARKET"),
    ("PERSON", "BORN_IN", "LOCATION"),
    ("PERSON", "BORN_IN", "TIME"),
    ("PERSON", "DIED_IN", "LOCATION"),
    ("PERSON", "DIED_IN", "TIME"),
    ("PERSON", "WORKED_ON", "EVENT"),
    ("PERSON", "WORKED_ON", "PRODUCT"),
    ("PERSON", "WORKED_ON", "CONCEPT"),
    ("PERSON", "WORKED_ON", "TECHNOLOGY"),
    ("LOCATION", "LOCATED_IN", "LOCATION"),
    ("LOCATION", "PART_OF", "LOCATION"),
]

DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT = PromptTemplate(
    "Give the following text, extract the knowledge graph according to the provided schema. "
    "Try to limit to the output {max_triplets_per_chunk} extracted paths.s\n"
    "-------\n"
    "{text}\n"
    "-------\n"
)


class SchemaLLMPathExtractor(TransformComponent):
    """
    Extract paths from a graph using a schema.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[PromptTemplate, str], optional):
            The template to use for the extraction query. Defaults to None.
        possible_entities (Optional[TypeAlias], optional):
            The possible entities to extract. Defaults to None.
        possible_relations (Optional[TypeAlias], optional):
            The possible relations to extract. Defaults to None.
        strict (bool, optional):
            Whether to enforce strict validation of entities and relations. Defaults to True.
            If false, values outside of the schema will be allowed.
        kg_schema_cls (Any, optional):
            The schema class to use. Defaults to None.
        kg_validation_schema (Dict[str, str], optional):
            The validation schema to use. Defaults to None.
        max_triplets_per_chunk (int, optional):
            The maximum number of triplets to extract per chunk. Defaults to 10.
        num_workers (int, optional):
            The number of workers to use. Defaults to 4.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    kg_schema_cls: Any
    kg_validation_schema: Dict[str, Any]
    num_workers: int
    max_triplets_per_chunk: int
    strict: bool

    def __init__(
        self,
        llm: LLM,
        extract_prompt: Union[PromptTemplate, str] = None,
        possible_entities: Optional[TypeAlias] = None,
        possible_relations: Optional[TypeAlias] = None,
        strict: bool = True,
        kg_schema_cls: Any = None,
        kg_validation_schema: Union[Dict[str, str], List[Triple]] = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        # Build a pydantic model on the fly
        if kg_schema_cls is None:
            possible_entities = possible_entities or DEFAULT_ENTITIES
            entity_cls = create_model(
                "Entity",
                type=(
                    possible_entities if strict else str,
                    Field(
                        ...,
                        description=(
                            "Entity in a knowledge graph. Only extract entities with types that are listed as valid: "
                            + str(possible_entities)
                        ),
                    ),
                ),
                name=(str, ...),
            )

            possible_relations = possible_relations or DEFAULT_RELATIONS
            relation_cls = create_model(
                "Relation",
                type=(
                    possible_relations if strict else str,
                    Field(
                        ...,
                        description=(
                            "Relation in a knowledge graph. Only extract relations with types that are listed as valid: "
                            + str(possible_relations)
                        ),
                    ),
                ),
            )

            triplet_cls = create_model(
                "Triplet",
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
                __validators__={"validator1": root},
                triplets=(List[triplet_cls], ...),
            )
            kg_schema_cls.__doc__ = "Knowledge Graph Schema."

        # Get validation schema
        kg_validation_schema = kg_validation_schema or DEFAULT_VALIDATION_SCHEMA
        # TODO: Remove this in a future version & encourage List[Triple] for validation schema
        if isinstance(kg_validation_schema, list):
            kg_validation_schema = {"relationships": kg_validation_schema}

        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt or DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT,
            kg_schema_cls=kg_schema_cls,
            kg_validation_schema=kg_validation_schema,
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            strict=strict,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SchemaLLMPathExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triplets from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    def _prune_invalid_triplets(self, kg_schema: Any) -> List[Triplet]:
        """Prune invalid triplets."""
        assert isinstance(kg_schema, self.kg_schema_cls)

        valid_triplets = []
        for triplet in kg_schema.triplets:
            subject = triplet.subject.name
            subject_type = triplet.subject.type

            relation = triplet.relation.type

            obj = triplet.object.name
            obj_type = triplet.object.type

            # Check if the triplet is valid based on the schema format
            if (
                isinstance(self.kg_validation_schema, dict)
                and "relationships" in self.kg_validation_schema
            ):
                # Schema is a dictionary with a 'relationships' key and triples as values
                if (subject_type, relation, obj_type) not in self.kg_validation_schema[
                    "relationships"
                ]:
                    continue
            else:
                # Schema is the backwards-compat format
                if relation not in self.kg_validation_schema.get(
                    subject_type, [relation]
                ) and relation not in self.kg_validation_schema.get(
                    obj_type, [relation]
                ):
                    continue

            # Remove self-references
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

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triplets:
            subj.properties = metadata
            obj.properties = metadata
            rel.properties = metadata

            existing_relations.append(rel)
            existing_nodes.append(subj)
            existing_nodes.append(obj)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triplets from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text with schema",
        )
