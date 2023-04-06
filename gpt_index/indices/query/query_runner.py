"""Query runner."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from gpt_index.data_structs.data_structs_v2 import CompositeIndex
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct as IndexStruct
from gpt_index.data_structs.node_v2 import IndexNode, Node, NodeWithScore
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.docstore import DocumentStore
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.query_combiner.base import (
    BaseQueryCombiner,
    get_default_query_combiner,
)
from gpt_index.indices.query.query_transform.base import (
    BaseQueryTransform,
    IdentityQueryTransform,
)
from gpt_index.indices.query.schema import QueryBundle, QueryConfig, QueryMode
from gpt_index.indices.service_context import ServiceContext
from gpt_index.response.schema import RESPONSE_TYPE

# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]

logger = logging.getLogger(__name__)


@dataclass
class QueryConfigMap:
    type_to_config_dict: Dict[str, QueryConfig]
    id_to_config_dict: Dict[str, QueryConfig]

    def get(self, index_struct: IndexStruct) -> QueryConfig:
        """Get query config."""
        index_struct_id = index_struct.index_id
        index_struct_type = index_struct.get_type()
        if index_struct_id in self.id_to_config_dict:
            config = self.id_to_config_dict[index_struct_id]
        elif index_struct_type in self.type_to_config_dict:
            config = self.type_to_config_dict[index_struct_type]
        else:
            config = QueryConfig(
                index_struct_type=index_struct_type, query_mode=QueryMode.DEFAULT
            )
        return config


def _get_query_config_map(
    query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
) -> QueryConfigMap:
    """Parse query config dicts."""
    type_to_config_dict: Dict[str, QueryConfig] = {}
    id_to_config_dict: Dict[str, QueryConfig] = {}
    if query_configs is None or len(query_configs) == 0:
        query_config_objs: List[QueryConfig] = []
    elif isinstance(query_configs[0], Dict):
        query_config_objs = [
            QueryConfig.from_dict(cast(Dict, qc)) for qc in query_configs
        ]
    else:
        query_config_objs = [cast(QueryConfig, q) for q in query_configs]

    for qc in query_config_objs:
        type_to_config_dict[qc.index_struct_type] = qc
        if qc.index_struct_id is not None:
            id_to_config_dict[qc.index_struct_id] = qc

    return QueryConfigMap(type_to_config_dict, id_to_config_dict)


class QueryRunner:
    """Tool to take in a query request and perform a query with the right classes.

    Higher-level wrapper over a given query.

    """

    def __init__(
        self,
        index_struct: IndexStruct,
        service_context: ServiceContext,
        docstore: DocumentStore,
        query_context: Dict[str, Dict[str, Any]],
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
        query_combiner: Optional[BaseQueryCombiner] = None,
        recursive: bool = False,
        use_async: bool = False,
    ) -> None:
        """Init params."""
        # data and services
        self._index_struct = index_struct
        self._service_context = service_context
        self._docstore = docstore
        self._query_context = query_context

        # query configurations and transformation
        self._query_config_map = _get_query_config_map(query_configs)
        self._query_transform = query_transform or IdentityQueryTransform()
        self._query_combiner = query_combiner

        # additional configs
        self._recursive = recursive
        self._use_async = use_async

    def _get_query_kwargs(self, config: QueryConfig) -> Dict[str, Any]:
        """Get query kwargs.

        Also update with default arguments if not present.

        """
        query_kwargs = {k: v for k, v in config.query_kwargs.items()}
        if "service_context" not in query_kwargs:
            query_kwargs["service_context"] = self._service_context
        return query_kwargs

    def _get_query_transform(self, index_struct: IndexStruct) -> BaseQueryTransform:
        """Get query transform."""
        config = self._query_config_map.get(index_struct)
        if config.query_transform is not None:
            query_transform = cast(BaseQueryTransform, config.query_transform)
        else:
            query_transform = self._query_transform
        return query_transform

    def _get_query_combiner(
        self, index_struct: IndexStruct, query_transform: BaseQueryTransform
    ) -> BaseQueryCombiner:
        """Get query transform."""
        config = self._query_config_map.get(index_struct)
        if config.query_combiner is not None:
            query_combiner: Optional[BaseQueryCombiner] = cast(
                BaseQueryCombiner, config.query_combiner
            )
        else:
            query_combiner = self._query_combiner

        # if query_combiner is still None, use default
        if query_combiner is None:
            extra_kwargs = {
                "service_context": self._service_context,
            }
            query_combiner = get_default_query_combiner(
                index_struct, query_transform, self, extra_kwargs=extra_kwargs
            )

        return cast(BaseQueryCombiner, query_combiner)

    def _get_query_obj(
        self,
        index_struct: IndexStruct,
    ) -> BaseGPTIndexQuery:
        """Get query object."""
        index_struct_type = index_struct.get_type()
        if index_struct_type == IndexStructType.COMPOSITE:
            raise ValueError("Cannot get query object for composite index struct.")
        config = self._query_config_map.get(index_struct)
        mode = config.query_mode

        from gpt_index.indices.registry import INDEX_STRUT_TYPE_TO_QUERY_MAP

        query_cls = INDEX_STRUT_TYPE_TO_QUERY_MAP[index_struct_type][mode]
        query_kwargs = self._get_query_kwargs(config)

        # Inject additional query context into query kwargs
        query_context = self._query_context.get(index_struct.index_id, {})
        query_kwargs.update(query_context)

        query_obj = query_cls(
            index_struct=index_struct,
            docstore=self._docstore,
            **query_kwargs,
        )

        return query_obj

    def query_transformed(
        self,
        query_bundle: QueryBundle,
        index_struct: V2IndexStruct,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        """This is called via BaseQueryCombiner.run."""
        query_obj = self._get_query_obj(index_struct)
        if self._recursive:
            logger.debug(f"> Query level : {level} on {index_struct.get_type()}")
            # call recursively
            nodes = query_obj.retrieve(query_bundle)

            # do recursion here
            nodes_for_synthesis = []
            additional_source_nodes = []
            for node_with_score in nodes:
                node_with_score, source_nodes = self._fetch_recursive_nodes(
                    node_with_score, query_bundle, level
                )
                nodes_for_synthesis.append(node_with_score)
                additional_source_nodes.extend(source_nodes)
            response = query_obj.synthesize(
                query_bundle, nodes_for_synthesis, additional_source_nodes
            )
            return response
        else:
            return query_obj.query(query_bundle)

    def _fetch_recursive_nodes(
        self,
        node_with_score: NodeWithScore,
        query_bundle: QueryBundle,
        level: int,
    ) -> Tuple[NodeWithScore, List[NodeWithScore]]:
        """Fetch nodes.

        Uses existing node if it's not an index node.
        Otherwise fetch response from corresponding index.

        """
        if isinstance(node_with_score.node, IndexNode):
            index_node = node_with_score.node
            # recursive call
            response = self.query(query_bundle, index_node.index_id, level + 1)

            new_node = Node(text=str(response))
            new_node_with_score = NodeWithScore(
                node=new_node, score=node_with_score.score
            )
            return new_node_with_score, response.source_nodes
        else:
            return node_with_score, []

    async def _afetch_recursive_nodes(
        self,
        node_with_score: NodeWithScore,
        query_bundle: QueryBundle,
        level: int,
    ) -> Tuple[NodeWithScore, List[NodeWithScore]]:
        """Fetch nodes.

        Usees existing node if it's not an index node.
        Otherwise fetch response from corresponding index.

        """
        if isinstance(node_with_score.node, IndexNode):
            index_node = node_with_score.node
            # recursive call
            response = await self.aquery(query_bundle, index_node.index_id, level + 1)

            new_node = Node(text=str(response))
            new_node_with_score = NodeWithScore(
                node=new_node, score=node_with_score.score
            )
            return new_node_with_score, response.source_nodes
        else:
            return node_with_score, []

    async def aquery_transformed(
        self,
        query_bundle: QueryBundle,
        index_struct: V2IndexStruct,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        """This is called via BaseQueryCombiner.run."""
        query_obj = self._get_query_obj(index_struct)
        if self._recursive:
            logger.debug(f"> Query level : {level} on {index_struct.get_type()}")
            # call recursively
            nodes = query_obj.retrieve(query_bundle)

            # do recursion here
            tasks = []
            nodes_for_synthesis = []
            additional_source_nodes = []

            for node_with_score in nodes:
                tasks.append(
                    self._afetch_recursive_nodes(node_with_score, query_bundle, level)
                )

            tuples = await asyncio.gather(*tasks)
            for node_with_score, source_nodes in tuples:
                nodes_for_synthesis.append(node_with_score)
                additional_source_nodes.extend(source_nodes)

            return await query_obj.asynthesize(
                query_bundle, nodes_for_synthesis, additional_source_nodes
            )
        else:
            return await query_obj.aquery(query_bundle)

    def _prepare_query_objects(
        self,
        query_str_or_bundle: Union[str, QueryBundle],
        index_id: Optional[str] = None,
    ) -> Tuple[BaseQueryCombiner, QueryBundle]:
        """Prepare query combiner and query bundle for query call."""
        # Resolve index struct from index_id if necessary
        if isinstance(self._index_struct, CompositeIndex):
            if index_id is None:
                index_id = self._index_struct.root_id
            assert index_id is not None
            index_struct = self._index_struct.all_index_structs[index_id]
        else:
            if index_id is not None:
                raise ValueError("index_id should be used with composite graph")
            index_struct = self._index_struct

        # Wrap query string as QueryBundle if necessary
        if isinstance(query_str_or_bundle, str):
            query_bundle = QueryBundle(
                query_str=query_str_or_bundle,
                custom_embedding_strs=[query_str_or_bundle],
            )
        else:
            query_bundle = query_str_or_bundle

        query_transform = self._get_query_transform(index_struct)
        query_combiner = self._get_query_combiner(index_struct, query_transform)
        return query_combiner, query_bundle

    def query(
        self,
        query_str_or_bundle: Union[str, QueryBundle],
        index_id: Optional[str] = None,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        """Run query.

        NOTE: Relies on mutual recursion between
            - QueryRunner.query
            - QueryRunner.query_transformed
            - BaseQueryCombiner.run

        QueryRunner.query resolves the current index struct,
            then pass control to BaseQueryCombiner.run.
        BaseQueryCombiner.run applies query transforms and makes multiple queries
            on the same index.
        Each query is passed to QueryRunner.query_transformed for execution.
            During execution, we recursively calls QueryRunner.query if index is a
            composable graph.
        """
        query_combiner, query_bundle = self._prepare_query_objects(
            query_str_or_bundle, index_id=index_id
        )
        return query_combiner.run(query_bundle, level)

    async def aquery(
        self,
        query_str_or_bundle: Union[str, QueryBundle],
        index_id: Optional[str] = None,
        level: int = 0,
    ) -> RESPONSE_TYPE:
        """Run query."""
        query_combiner, query_bundle = self._prepare_query_objects(
            query_str_or_bundle, index_id=index_id
        )
        return await query_combiner.arun(query_bundle, level)
