import asyncio
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import create_model, field_validator
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    Triplet,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.indices.property_graph.transformations.utils import (
    get_entity_class,
    get_relation_class,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
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
        possible_entities (Optional[Type[Any]], optional):
            The possible entities to extract. Defaults to None.
        possible_entity_props (Optional[Union[List[str], List[Tuple[str, str]]], optional):
            The possible entity properties to extract. Defaults to None.
            Can be a list of strings or a list of tuples with the format (name, description).
        possible_relations (Optional[Type[Any]], optional):
            The possible relations to extract. Defaults to None.
        possible_relation_props (Optional[Union[List[str], List[Tuple[str, str]]], optional):
            The possible relation properties to extract. Defaults to None.
            Can be a list of strings or a list of tuples with the format (name, description).
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
    possible_entity_props: Optional[List[str]]
    possible_relation_props: Optional[List[str]]
    strict: bool

    def __init__(
        self,
        llm: LLM,
        extract_prompt: Optional[Union[PromptTemplate, str]] = None,
        possible_entities: Optional[Type[Any]] = None,
        possible_entity_props: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
        possible_relations: Optional[Type[Any]] = None,
        possible_relation_props: Optional[
            Union[List[str], List[Tuple[str, str]]]
        ] = None,
        strict: bool = True,
        kg_schema_cls: Any = None,
        kg_validation_schema: Optional[Union[Dict[str, str], List[Triple]]] = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        # Build a pydantic model on the fly
        if kg_schema_cls is None:
            possible_entities = possible_entities or DEFAULT_ENTITIES  # type: ignore
            if possible_entity_props and isinstance(possible_entity_props[0], tuple):
                entity_props = [  # type: ignore
                    f"Property label `{k}` with description ({v})"
                    for k, v in possible_entity_props
                ]
            else:
                entity_props = possible_entity_props  # type: ignore
            entity_cls = get_entity_class(possible_entities, entity_props, strict)

            possible_relations = possible_relations or DEFAULT_RELATIONS  # type: ignore
            if possible_relation_props and isinstance(
                possible_relation_props[0], tuple
            ):
                relation_props = [  # type: ignore
                    f"Property label `{k}` with description ({v})"
                    for k, v in possible_relation_props
                ]
            else:
                relation_props = possible_relation_props  # type: ignore
            relation_cls = get_relation_class(
                possible_relations, relation_props, strict
            )

            triplet_cls = create_model(
                "Triplet",
                subject=(entity_cls, ...),
                relation=(relation_cls, ...),
                object=(entity_cls, ...),
            )

            def validate(v: Any) -> Any:
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

            root = field_validator("triplets", mode="before")(validate)
            kg_schema_cls = create_model(
                "KGSchema",
                __validators__={"validator1": root},  # type: ignore
                triplets=(List[triplet_cls], ...),  # type: ignore
            )
            kg_schema_cls.__doc__ = "Knowledge Graph Schema."

        # Get validation schema
        kg_validation_schema = kg_validation_schema or DEFAULT_VALIDATION_SCHEMA
        # TODO: Remove this in a future version & encourage List[Triple] for validation schema
        if isinstance(kg_validation_schema, list):
            kg_validation_schema = {"relationships": kg_validation_schema}  # type: ignore

        # flatten tuples now that we don't need the descriptions
        if possible_relation_props and isinstance(possible_relation_props[0], tuple):
            possible_relation_props = [x[0] for x in possible_relation_props]

        if possible_entity_props and isinstance(possible_entity_props[0], tuple):
            possible_entity_props = [x[0] for x in possible_entity_props]

        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt or DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT,
            kg_schema_cls=kg_schema_cls,
            kg_validation_schema=kg_validation_schema,
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            possible_entity_props=possible_entity_props,
            possible_relation_props=possible_relation_props,
            strict=strict,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SchemaLLMPathExtractor"

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triplets from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    def _prune_invalid_props(
        self, props: Dict[str, Any], allowed_props: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Prune invalid properties."""
        if not allowed_props:
            return props

        props_to_remove = []
        for key in props:
            if key not in allowed_props:
                props_to_remove.append(key)

        for key in props_to_remove:
            del props[key]

        return props

    def _prune_invalid_triplets(self, kg_schema: Any) -> Sequence[Triplet]:
        """Prune invalid triplets."""
        valid_triplets = []
        for triplet in kg_schema.triplets:
            subject = triplet.subject.name
            subject_type = triplet.subject.type
            subject_props: Dict[str, Any] = {}
            if hasattr(triplet.subject, "properties"):
                subject_props = triplet.subject.properties or {}
                if self.strict:
                    subject_props = self._prune_invalid_props(
                        subject_props,
                        self.possible_entity_props,
                    )

            relation = triplet.relation.type
            relation_props: Dict[str, Any] = {}
            if hasattr(triplet.relation, "properties"):
                relation_props = triplet.relation.properties or {}
                if self.strict:
                    relation_props = self._prune_invalid_props(
                        relation_props,
                        self.possible_relation_props,
                    )

            obj = triplet.object.name
            obj_type = triplet.object.type
            obj_props: Dict[str, Any] = {}
            if hasattr(triplet.object, "properties"):
                obj_props = triplet.object.properties or {}
                if self.strict:
                    obj_props = self._prune_invalid_props(
                        obj_props,
                        self.possible_entity_props,
                    )

            # Check if the triplet is valid based on the schema format
            if self.strict:
                if (
                    isinstance(self.kg_validation_schema, dict)
                    and "relationships" in self.kg_validation_schema
                ):
                    # Schema is a dictionary with a 'relationships' key and triples as values
                    if (
                        subject_type,
                        relation,
                        obj_type,
                    ) not in self.kg_validation_schema["relationships"]:
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

            subj_node = EntityNode(
                label=subject_type, name=subject, properties=subject_props
            )
            obj_node = EntityNode(label=obj_type, name=obj, properties=obj_props)
            rel_node = Relation(
                label=relation,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=relation_props,
            )
            valid_triplets.append((subj_node, rel_node, obj_node))

        return valid_triplets

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triplets from a node."""
        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            kg_schema = await self.llm.astructured_predict(
                self.kg_schema_cls,
                self.extract_prompt,
                text=text,
                max_triplets_per_chunk=self.max_triplets_per_chunk,
            )
            triplets = self._prune_invalid_triplets(kg_schema)
        except (ValueError, TypeError, AttributeError):
            triplets = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triplets:
            subj.properties.update(metadata)
            obj.properties.update(metadata)
            rel.properties.update(metadata)

            existing_relations.append(rel)
            existing_nodes.append(subj)
            existing_nodes.append(obj)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
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
