"""NebulaGraph graph store index."""
import logging
import os
from typing import Any, Dict, List, Optional

from llama_index.graph_stores.types import GraphStore

QUOTE = '"'
RETRY_TIMES = 3

logger = logging.getLogger(__name__)


def hash_string_to_rank(string: str) -> int:
    # get signed 64-bit hash value
    signed_hash = hash(string)

    # reduce the hash value to a 64-bit range
    mask = (1 << 64) - 1
    signed_hash &= mask

    # convert the signed hash value to an unsigned 64-bit integer
    if signed_hash & (1 << 63):
        unsigned_hash = -((signed_hash ^ mask) + 1)
    else:
        unsigned_hash = signed_hash

    return unsigned_hash


def prepare_subjs_param(subjs: Optional[List[str]]) -> dict:
    """Prepare parameters for query."""
    if subjs is None:
        return {}
    from nebula3.common import ttypes

    subjs_list = []
    subjs_byte = ttypes.Value()
    for subj in subjs:
        if not isinstance(subj, str):
            raise TypeError(f"Subject should be str, but got {type(subj).__name__}.")
        subj_byte = ttypes.Value()
        subj_byte.set_sVal(subj)
        subjs_list.append(subj_byte)
    subjs_nlist = ttypes.NList(values=subjs_list)
    subjs_byte.set_lVal(subjs_nlist)
    return {"subjs": subjs_byte}


class NebulaGraphStore(GraphStore):
    """NebulaGraph graph store."""

    def __init__(
        self,
        session_pool: Optional[Any] = None,
        space_name: Optional[str] = None,
        edge_types: Optional[List[str]] = ["rel"],
        rel_prop_names: Optional[List[str]] = ["predicate"],
        tags: Optional[List[str]] = ["entity"],
        session_pool_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs: Any,
    ) -> None:
        """Initialize NebulaGraph graph store.

        Args:
            session_pool: NebulaGraph session pool.
            space_name: NebulaGraph space name.
            edge_types: Edge types.
            rel_prop_names: Relation property names corresponding to edge types.
            session_pool_kwargs: Keyword arguments for NebulaGraph session pool.
            **kwargs: Keyword arguments.
        """
        try:
            import nebula3  # noqa
        except ImportError:
            raise ImportError(
                "Please install NebulaGraph Python client first: "
                "`pip install nebula3-python`"
            )
        assert space_name is not None, "space_name should be provided."
        self._space_name = space_name
        self._session_pool_kwargs = session_pool_kwargs
        if (
            self._session_pool_kwargs is not None
            and "retry" in self._session_pool_kwargs
        ):
            self._retry = self._session_pool_kwargs.pop("retry")
        else:
            self._retry = RETRY_TIMES

        if session_pool is None:
            self.init_session_pool()

        self._tags = tags or ["entity"]
        self._edge_types = edge_types or ["rel"]
        self._rel_prop_names = rel_prop_names or ["predicate"]
        if len(self._edge_types) != len(self._rel_prop_names):
            raise ValueError(
                "edge_types and rel_prop_names to define relation and relation name"
                "should be provided."
            )
        if len(self._edge_types) == 0:
            raise ValueError("Length of `edge_types` should be greater than 0.")

        # for building query
        self._edge_dot_rel = [
            f"`{edge_type}`.`{rel_prop_name}`"
            for edge_type, rel_prop_name in zip(self._edge_types, self._rel_prop_names)
        ]

    def init_session_pool(self) -> Any:
        """Return NebulaGraph session pool."""
        from nebula3.Config import SessionPoolConfig
        from nebula3.gclient.net.SessionPool import SessionPool

        # ensure "NEBULA_USER", "NEBULA_PASSWORD", "NEBULA_ADDRESS" are set
        # in environment variables
        if not all(
            key in os.environ
            for key in ["NEBULA_USER", "NEBULA_PASSWORD", "NEBULA_ADDRESS"]
        ):
            raise ValueError(
                "NEBULA_USER, NEBULA_PASSWORD, NEBULA_ADDRESS should be set in "
                "environment variables when NebulaGraph Session Pool is not "
                "directly passed."
            )
        graphd_host, graphd_port = os.environ["NEBULA_ADDRESS"].split(":")
        session_pool = SessionPool(
            os.environ["NEBULA_USER"],
            os.environ["NEBULA_PASSWORD"],
            self._space_name,
            [(graphd_host, int(graphd_port))],
        )

        seesion_pool_config = SessionPoolConfig()
        session_pool.init(seesion_pool_config)
        self._session_pool = session_pool
        return self._session_pool

    def __del__(self) -> None:
        """Close NebulaGraph session pool."""
        self._session_pool.close()

    def execute(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Execute query.

        Args:
            query: Query.
            param_map: Parameter map.

        Returns:
            Query result.
        """
        from nebula3.Exception import IOErrorException
        from nebula3.fbthrift.transport.TTransport import TTransportException

        retry = self._retry
        while retry > 0:
            try:
                result = self._session_pool.execute_parameter(query, param_map)
                if not result.is_succeeded():
                    raise ValueError(result.error_msg())
                return result
            except (TTransportException, IOErrorException) as e:
                # connection issue, try to recreate session pool
                if retry > 0:
                    retry -= 2
                    # try to recreate session pool
                    self.init_session_pool()
                else:
                    raise e
            except ValueError as e:
                # query failed on db side
                if retry > 0:
                    retry -= 1
                    continue
                else:
                    raise e
            except Exception as e:
                raise e

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)

    @property
    def client(self) -> Any:
        """Return NebulaGraph session pool."""
        return self._session_pool

    @property
    def config_dict(self) -> dict:
        """Return configuration dictionary."""
        return {
            "session_pool": self._session_pool,
            "space_name": self._space_name,
            "edge_types": self._edge_types,
            "rel_prop_names": self._rel_prop_names,
            "session_pool_kwargs": self._session_pool_kwargs,
        }

    def get(self, sub: str) -> List[List[str]]:
        """Get triplets.

        Args:
            sub: Subject.

        Returns:
            Triplets.
        """
        if len(self._edge_types) == 1:
            # edge_types = ["follow"]
            # rel_prop_names = ["degree"]
            # GO FROM "player100" OVER `follow``
            # YIELD `follow`.`degree`` AS rel, dst(edge) AS obj
            query = (
                f"GO FROM {QUOTE}{sub}{QUOTE} OVER `{self._edge_types[0]}`"
                f"YIELD `{self._edge_types[0]}`.`{self._rel_prop_names[0]}` AS rel, "
                f"dst(edge) AS obj"
            )
        else:
            # edge_types = ["follow", "serve"]
            # rel_prop_names = ["degree", "start_year"]
            # GO FROM "player100" OVER `follow`, `serve`
            # YIELD [value IN [follow.degree,serve.start_year]
            # WHERE value IS NOT EMPTY ][0] AS rel, dst(edge) AS obj
            query = (
                f"GO FROM {QUOTE}{sub}{QUOTE} OVER "
                f"`{'`, `'.join(self._edge_types)}` "
                f"YIELD "
                f"[value IN [{', '.join(self._edge_dot_rel)}] "
                f"WHERE value IS NOT EMPTY][0] AS rel, "
                f"dst(edge) AS obj"
            )
        logger.debug(f"Query: {query}")
        result = self.execute(query)

        # get raw data
        rels = result.column_values("rel")
        objs = result.column_values("obj")

        # convert to list of list
        return [[str(rel.cast()), str(obj.cast())] for rel, obj in zip(rels, objs)]

    def get_flat_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: sub -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+
        if len(self._edge_types) == 1:
            # MATCH (s)-[e:follow*..2]->() WHERE id(s) IN ["player100", "player101"]
            #   WITH id(s) AS subj, [rel in e | [rel.degree, dst(rel)] ] AS rels
            # RETURN
            #   subj,
            #   REDUCE(acc = collect(NULL), l in rels | acc + l) AS flattened_rels
            query = (
                f"MATCH (s)-[e:`{self._edge_types[0]}`*..{depth}]->() "
                f"  WHERE id(s) IN $subjs "
                f"WITH "
                f"id(s) AS subj,"
                f"[rel IN e | "
                f"  [rel.`{self._rel_prop_names[0]}`, dst(rel)] "
                f"] AS rels "
                f"RETURN "
                f"  subj,"
                f"  REDUCE(acc = collect(NULL), l in rels | acc + l)"
                f"    AS flattened_rels"
            )
        else:
            # edge_types = ["follow", "serve"]
            # rel_prop_names = ["degree", "start_year"]
            # MATCH (s)-[e:follow|serve*..2]->()
            # WHERE id(s) IN ["player100", "player101"]
            #   WITH id(s) AS subj,
            # [rel in e | [CASE type(rel)
            #     WHEN "follow" THEN rel.degree
            #     WHEN "serve" THEN rel.start_year
            #     END, dst(rel)] ]
            #     AS rels
            # RETURN
            #   subj,
            #   REDUCE(acc = collect(NULL), l in rels | acc + l) AS flattened_rels
            _case_when_string = "".join(
                [
                    f"WHEN {QUOTE}{edge_type}{QUOTE} THEN rel.`{rel_prop_name}` "
                    for edge_type, rel_prop_name in zip(
                        self._edge_types, self._rel_prop_names
                    )
                ]
            )
            query = (
                f"MATCH (s)-[e:`{'`|`'.join(self._edge_types)}`*..{depth}]-() "
                f"  WHERE id(s) IN $subjs "
                f"WITH "
                f"id(s) AS subj,"
                f"[rel IN e | "
                f"  [CASE type(rel) "
                f"  {_case_when_string}"
                f"  END, dst(rel)] "
                f"] AS rels "
                f"RETURN"
                f"  subj,"
                f"  REDUCE(acc = collect(NULL), l in rels | acc + l) AS flattened_rels"
            )
        subjs_param = prepare_subjs_param(subjs)
        logger.debug(f"get_flat_rel_map() subjs_param: {subjs}, query: {query}")
        result = self.execute(query, subjs_param)

        # get raw data
        subjs_ = result.column_values("subj") or []
        rels_ = result.column_values("flattened_rels") or []

        rel_map: Dict[Any, List[Any]] = {}
        for sub, rel in zip(subjs_, rels_):
            sub_ = sub.cast()
            rel_ = rel.cast()
            if sub_ not in rel_map:
                rel_map[sub_] = []
            rel_map[sub_].append(rel_)
        return rel_map

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get rel map."""
        # We put rels in a long list for depth>= 1, this is different from
        # SimpleGraphStore.get_rel_map() though.
        # But this makes more sense for multi-hop relation path.

        # lower case subjs
        subjs = [sub.lower() for sub in subjs] if subjs else None

        return self.get_flat_rel_map(subjs, depth)

    def upsert_triplet(self, sub: str, rel: str, obj: str) -> None:
        """Add triplet."""
        # Note, to enable leveraging existing knowledge graph,
        # the (triplet -- property graph) mapping
        #   makes (n:1) edge_type.prop_name --> triplet.rel
        # thus we have to assume rel to be the first edge_type.prop_name
        # here in upsert_triplet().
        # This applies to the type of entity(tags) with subject and object, too,
        # thus we have to assume sub to be the first entity.tag_name

        # lower case sub, rel, obj
        sub = sub.lower()
        rel = rel.lower()
        obj = obj.lower()

        edge_type = self._edge_types[0]
        rel_prop_name = self._rel_prop_names[0]
        entity_type = self._tags[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"INSERT VERTEX `{entity_type}`() "
            f"  VALUES {QUOTE}{sub}{QUOTE}:();"
            f"INSERT VERTEX `{entity_type}`() "
            f"  VALUES {QUOTE}{obj}{QUOTE}:();"
            f"INSERT EDGE `{edge_type}`(`{rel_prop_name}`) "
            f"  VALUES "
            f"{QUOTE}{sub}{QUOTE}->{QUOTE}{obj}{QUOTE}"
            f"@{rel_hash}:({QUOTE}{rel}{QUOTE});"
        )
        logger.debug(f"upsert_triplet() DML query: {dml_query}")
        result = self.execute(dml_query)
        assert (
            result.is_succeeded()
        ), f"Failed to upsert triplet: {sub} {rel} {obj}, query: {dml_query}"

    def delete(self, sub: str, rel: str, obj: str) -> None:
        """Delete triplet.
        1. Similar to upsert_triplet(),
           we have to assume rel to be the first edge_type.prop_name.
        2. After edge being deleted, we need to check if the sub or
           obj are isolated vertices,
           if so, delete them, too.
        """

        # lower case sub, rel, obj
        sub = sub.lower()
        rel = rel.lower()
        obj = obj.lower()

        # DELETE EDGE serve "player100" -> "team204"@7696463696635583936;
        edge_type = self._edge_types[0]
        # rel_prop_name = self._rel_prop_names[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"DELETE EDGE `{edge_type}`"
            f"  {QUOTE}{sub}{QUOTE}->{QUOTE}{obj}{QUOTE}@{rel_hash};"
        )
        logger.debug(f"delete() DML query: {dml_query}")
        result = self.execute(dml_query)
        assert (
            result.is_succeeded()
        ), f"Failed to delete triplet: {sub} {rel} {obj}, query: {dml_query}"
        # Get isolated vertices to be deleted
        # MATCH (s) WHERE id(s) IN ["player700"] AND NOT (s)-[]-()
        # RETURN id(s) AS isolated
        query = (
            f"MATCH (s) "
            f"  WHERE id(s) IN [{QUOTE}{sub}{QUOTE}, {QUOTE}{obj}{QUOTE}] "
            f"  AND NOT (s)-[]-() "
            f"RETURN id(s) AS isolated"
        )
        result = self.execute(query)
        isolated = result.column_values("isolated")
        if not isolated:
            return
        # DELETE VERTEX "player700"
        vertex_ids = ",".join([f"{QUOTE}{v.cast()}{QUOTE}" for v in isolated])
        dml_query = f"DELETE VERTEX {vertex_ids};"

        result = self.execute(dml_query)
        assert (
            result.is_succeeded()
        ), f"Failed to delete isolated vertices: {isolated}, query: {dml_query}"
