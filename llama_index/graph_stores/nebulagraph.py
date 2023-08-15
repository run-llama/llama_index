"""NebulaGraph graph store index."""
import logging
import os
from string import Template
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

from llama_index.graph_stores.types import GraphStore

QUOTE = '"'
RETRY_TIMES = 3
WAIT_MIN_SECONDS = 0.5
WAIT_MAX_SECONDS = 10

logger = logging.getLogger(__name__)


rel_query = Template(
    """
LOOKUP ON $edge_type 
YIELD src(edge) AS srcVid |
GO FROM $$-.srcVid OVER $edge_type 
        YIELD "(:" + tags($$^)[0] + ")-[:$edge_type]->(:" + tags($$$$)[0] + ")" AS rels | limit 1
"""
)


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


def escape_str(value: str) -> str:
    """Escape String for NebulaGraph Query."""
    patterns = {
        '"': " ",
    }
    for pattern in patterns:
        if pattern in value:
            value = value.replace(pattern, patterns[pattern])
    if value[0] == " " or value[-1] == " ":
        value = value.strip()

    return value


class NebulaGraphStore(GraphStore):
    """NebulaGraph graph store."""

    def __init__(
        self,
        session_pool: Optional[Any] = None,
        space_name: Optional[str] = None,
        edge_types: Optional[List[str]] = ["relationship"],
        rel_prop_names: Optional[List[str]] = ["relationship"],
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

    @retry(
        wait=wait_random_exponential(min=WAIT_MIN_SECONDS, max=WAIT_MAX_SECONDS),
        stop=stop_after_attempt(RETRY_TIMES),
    )
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

        # Clean the query string by removing triple backticks
        query = query.replace("```", "").strip()

        try:
            result = self._session_pool.execute_parameter(query, param_map)
            if result is None:
                raise ValueError(f"Query failed. Query: {query}, Param: {param_map}")
            if not result.is_succeeded():
                raise ValueError(
                    f"Query failed. Query: {query}, Param: {param_map}"
                    f"Error message: {result.error_msg()}"
                )
            return result
        except (TTransportException, IOErrorException, RuntimeError) as e:
            logger.error(
                f"Connection issue, try to recreate session pool. Query: {query}, "
                f"Param: {param_map}"
                f"Erorr: {e}"
            )
            self.init_session_pool()
            logger.info(
                f"Session pool recreated. Query: {query}, Param: {param_map}"
                f"This was due to error: {e}, and now retrying."
            )
            raise e

        except ValueError as e:
            # query failed on db side
            logger.error(
                f"Query failed. Query: {query}, Param: {param_map}"
                f"Error message: {e}"
            )
            raise e
        except Exception as e:
            # other exceptions
            logger.error(
                f"Query failed. Query: {query}, Param: {param_map}"
                f"Error message: {e}"
            )
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

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets.

        Args:
            subj: Subject.

        Returns:
            Triplets.
        """
        if len(self._edge_types) == 1:
            # edge_types = ["follow"]
            # rel_prop_names = ["degree"]
            # GO FROM "player100" OVER `follow``
            # YIELD `follow`.`degree`` AS rel, dst(edge) AS obj
            query = (
                f"GO FROM {QUOTE}{subj}{QUOTE} OVER `{self._edge_types[0]}`"
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
                f"GO FROM {QUOTE}{subj}{QUOTE} OVER "
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
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+
        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        if len(self._edge_types) == 1:
            # MATCH (s)-[e:follow*..2]-() WHERE id(s) IN ["player100", "player101"]
            #   WITH id(s) AS subj, [rel in e | [rel.degree, dst(rel)] ] AS rels
            #   WITH
            #       subj,
            #       REDUCE(acc = collect(NULL), l in rels | acc + l) AS flattened_rels
            # RETURN
            #   subj,
            #   REDUCE(acc = subj,l in flattened_rels|acc + ', ' + l) AS flattened_rels
            query = (
                f"MATCH (s)-[e:`{self._edge_types[0]}`*..{depth}]-() "
                f"  WHERE id(s) IN $subjs "
                f"WITH "
                f"id(s) AS subj,"
                f"[rel IN e | "
                f"  [rel.`{self._rel_prop_names[0]}`, dst(rel)] "
                f"] AS rels "
                f"WITH "
                f"  subj,"
                f"  REDUCE(acc = collect(NULL), l in rels | acc + l)"
                f"    AS flattened_rels"
                f" RETURN"
                f"  subj,"
                f"  REDUCE(acc = subj, l in flattened_rels | acc + ', ' + l )"
                f"    AS flattened_rels"
            )
        else:
            # edge_types = ["follow", "serve"]
            # rel_prop_names = ["degree", "start_year"]
            # MATCH (s)-[e:follow|serve*..2]-()
            # WHERE id(s) IN ["player100", "player101"]
            #   WITH id(s) AS subj,
            #        [rel in e | [CASE type(rel)
            #     WHEN "follow" THEN rel.degree
            #     WHEN "serve" THEN rel.start_year
            #     END, dst(rel)] ]
            #     AS rels
            #   WITH
            #     subj,
            #     REDUCE(acc = collect(NULL), l in rels | acc + l) AS
            #       flattened_rels
            # RETURN
            #   subj,
            #   REDUCE(acc = subj, l in flattened_rels | acc + ', ' + l ) AS
            #       flattened_rels
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
                f"  WITH "
                f"    id(s) AS subj,"
                f" [rel IN e | "
                f"  [CASE type(rel) "
                f"  {_case_when_string}"
                f"  END, dst(rel)] "
                f"] AS rels "
                f"  WITH"
                f"    subj,"
                f"    REDUCE(acc = collect(NULL), l in rels | acc + l) AS "
                f"        flattened_rels"
                f"RETURN"
                f"  subj,"
                f"  REDUCE(acc = subj, l in flattened_rels | acc + ', ' + l ) AS "
                f"      flattened_rels"
            )
        subjs_param = prepare_subjs_param(subjs)
        logger.debug(f"get_flat_rel_map() subjs_param: {subjs}, query: {query}")
        result = self.execute(query, subjs_param)
        if result is None:
            return rel_map

        # get raw data
        subjs_ = result.column_values("subj") or []
        rels_ = result.column_values("flattened_rels") or []

        for subj, rel in zip(subjs_, rels_):
            subj_ = subj.cast()
            rel_ = rel.cast()
            if subj_ not in rel_map:
                rel_map[subj_] = []
            rel_map[subj_].append(rel_)
        return rel_map

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get rel map."""
        # We put rels in a long list for depth>= 1, this is different from
        # SimpleGraphStore.get_rel_map() though.
        # But this makes more sense for multi-hop relation path.

        if subjs is not None:
            subjs = [escape_str(subj) for subj in subjs]
            if len(subjs) == 0:
                return {}

        return self.get_flat_rel_map(subjs, depth)

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        # Note, to enable leveraging existing knowledge graph,
        # the (triplet -- property graph) mapping
        #   makes (n:1) edge_type.prop_name --> triplet.rel
        # thus we have to assume rel to be the first edge_type.prop_name
        # here in upsert_triplet().
        # This applies to the type of entity(tags) with subject and object, too,
        # thus we have to assume subj to be the first entity.tag_name

        # lower case subj, rel, obj
        subj = escape_str(subj)
        rel = escape_str(rel)
        obj = escape_str(obj)

        edge_type = self._edge_types[0]
        rel_prop_name = self._rel_prop_names[0]
        entity_type = self._tags[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"INSERT VERTEX `{entity_type}`(name) "
            f"  VALUES {QUOTE}{subj}{QUOTE}:({QUOTE}{subj}{QUOTE});"
            f"INSERT VERTEX `{entity_type}`(name) "
            f"  VALUES {QUOTE}{obj}{QUOTE}:({QUOTE}{obj}{QUOTE});"
            f"INSERT EDGE `{edge_type}`(`{rel_prop_name}`) "
            f"  VALUES "
            f"{QUOTE}{subj}{QUOTE}->{QUOTE}{obj}{QUOTE}"
            f"@{rel_hash}:({QUOTE}{rel}{QUOTE});"
        )
        logger.debug(f"upsert_triplet() DML query: {dml_query}")
        result = self.execute(dml_query)
        assert (
            result and result.is_succeeded()
        ), f"Failed to upsert triplet: {subj} {rel} {obj}, query: {dml_query}"

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet.
        1. Similar to upsert_triplet(),
           we have to assume rel to be the first edge_type.prop_name.
        2. After edge being deleted, we need to check if the subj or
           obj are isolated vertices,
           if so, delete them, too.
        """

        # lower case subj, rel, obj
        subj = escape_str(subj)
        rel = escape_str(rel)
        obj = escape_str(obj)

        # DELETE EDGE serve "player100" -> "team204"@7696463696635583936;
        edge_type = self._edge_types[0]
        # rel_prop_name = self._rel_prop_names[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"DELETE EDGE `{edge_type}`"
            f"  {QUOTE}{subj}{QUOTE}->{QUOTE}{obj}{QUOTE}@{rel_hash};"
        )
        logger.debug(f"delete() DML query: {dml_query}")
        result = self.execute(dml_query)
        assert (
            result and result.is_succeeded()
        ), f"Failed to delete triplet: {subj} {rel} {obj}, query: {dml_query}"
        # Get isolated vertices to be deleted
        # MATCH (s) WHERE id(s) IN ["player700"] AND NOT (s)-[]-()
        # RETURN id(s) AS isolated
        query = (
            f"MATCH (s) "
            f"  WHERE id(s) IN [{QUOTE}{subj}{QUOTE}, {QUOTE}{obj}{QUOTE}] "
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
            result and result.is_succeeded()
        ), f"Failed to delete isolated vertices: {isolated}, query: {dml_query}"

    def refresh_schema(self) -> None:
        """
        Refreshes the NebulaGraph Store Schema.
        """
        tags_schema, edge_types_schema, relationships = [], [], []
        for tag in self.execute("SHOW TAGS").column_values("Name"):
            tag_name = tag.cast()
            tag_schema = {"tag": tag_name, "properties": []}
            r = self.execute(f"DESCRIBE TAG `{tag_name}`")
            props, types = r.column_values("Field"), r.column_values("Type")
            for i in range(r.row_size()):
                tag_schema["properties"].append((props[i].cast(), types[i].cast()))
            tags_schema.append(tag_schema)
        for edge_type in self.execute("SHOW EDGES").column_values("Name"):
            edge_type_name = edge_type.cast()
            edge_schema = {"edge": edge_type_name, "properties": []}
            r = self.execute(f"DESCRIBE EDGE `{edge_type_name}`")
            props, types = r.column_values("Field"), r.column_values("Type")
            for i in range(r.row_size()):
                edge_schema["properties"].append((props[i].cast(), types[i].cast()))
            edge_types_schema.append(edge_schema)

            # build relationships types
            r = self.execute(
                rel_query.substitute(edge_type=edge_type_name)
            ).column_values("rels")
            if len(r) > 0:
                relationships.append(r[0].cast())

        self.schema = (
            f"Node properties: {tags_schema}\n"
            f"Edge properties: {edge_types_schema}\n"
            f"Relationships: {relationships}\n"
        )

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the NebulaGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        result = self.execute(query, param_map)
        columns = result.keys()
        d: Dict[str, list] = {}
        for col_num in range(result.col_size()):
            col_name = columns[col_num]
            col_list = result.column_values(col_name)
            d[col_name] = [x.cast() for x in col_list]
        return d
