from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

from sqlalchemy import (
    Index,
    Column,
    Integer,
    UniqueConstraint,
    text,
    delete,
    select,
    create_engine,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from llama_index.core.llms import ChatMessage
from sqlalchemy.dialects.postgresql import JSON, ARRAY, JSONB, VARCHAR, insert
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore
from sqlalchemy import cast, bindparam


def get_data_model(
    base: type,
    index_name: str,
    schema_name: str,
    use_jsonb: bool = False,
) -> Any:
    """
    Create a dynamic SQLAlchemy model class for storing chat data in YugabyteDB.

    This function generates a SQLAlchemy model class with a table structure optimized for
    storing chat messages. The table includes columns for a unique key and an array of
    message values stored in either JSON or JSONB format.

    Args:
        base (type): The declarative base class from SQLAlchemy's `declarative_base()`.
        index_name (str): The base name to use for the table and class. Will be normalized
                         to create valid SQL identifiers (e.g., 'chat_store' becomes 'data_chat_store').
        schema_name (str): The database schema where the table will be created.
        use_jsonb (bool, optional): If True, uses JSONB column type for better query performance
                                   and indexing capabilities. If False, uses standard JSON.
                                   Defaults to False.

    """
    tablename = f"data_{index_name}"  # dynamic table name
    class_name = f"Data{index_name}"  # dynamic class name

    chat_dtype = JSONB if use_jsonb else JSON

    class AbstractData(base):  # type: ignore
        __abstract__ = True  # this line is necessary
        id = Column(Integer, primary_key=True, autoincrement=True)  # Add primary key
        key = Column(VARCHAR, nullable=False)
        value = Column(ARRAY(chat_dtype))

    return type(
        class_name,
        (AbstractData,),
        {
            "__tablename__": tablename,
            "__table_args__": (
                UniqueConstraint("key", name=f"{tablename}:unique_key"),
                Index(f"{tablename}:idx_key", "key"),
                {"schema": schema_name},
            ),
        },
    )


class YugabyteDBChatStore(BaseChatStore):
    table_name: Optional[str] = Field(
        default="chatstore", description="YugabyteDB table name."
    )
    schema_name: Optional[str] = Field(
        default="public", description="YugabyteDB schema name."
    )

    _table_class: Optional[Any] = PrivateAttr()
    _session: Optional[sessionmaker] = PrivateAttr()

    def __init__(
        self,
        session: sessionmaker,
        table_name: str,
        schema_name: str = "public",
        use_jsonb: bool = False,
    ):
        super().__init__(
            table_name=table_name.lower(),
            schema_name=schema_name.lower(),
        )

        # sqlalchemy model
        base = declarative_base()
        self._table_class = get_data_model(
            base,
            table_name,
            schema_name,
            use_jsonb=use_jsonb,
        )
        self._session = session
        self._initialize(base)

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        load_balance: Optional[bool] = False,
        topology_keys: Optional[str] = None,
        yb_servers_refresh_interval: Optional[int] = 300,
        fallback_to_topology_keys_only: Optional[bool] = False,
        failed_host_ttl_seconds: Optional[int] = 5,
        table_name: str = "chatstore",
        schema_name: str = "public",
        connection_string: Optional[str] = None,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> "YugabyteDBChatStore":
        """
        Return connection string from database parameters.

        Args:
            host (str): YugabyteDB host.
            port (str): YugabyteDB port.
            database (str): YugabyteDB database name.
            user (str): YugabyteDB user.
            password (str): YugabyteDB password.
            load_balance (bool, optional): Enables uniform load balancing. Defaults to False.
            topology_keys (str, optional): Enables topology-aware load balancing.
                Specify comma-separated geo-locations in the form of cloud.region.zone:priority.
                Ignored if load_balance is false. Defaults to None.
            yb_servers_refresh_interval (int, optional): The interval in seconds to refresh the servers list;
                ignored if load_balance is false. Defaults to 300.
            fallback_to_topology_keys_only (bool, optional): If set to true and topology_keys are specified,
                the driver only tries to connect to nodes specified in topology_keys
                Defaults to False.
            failed_host_ttl_seconds (int, optional): Time, in seconds, to wait before trying to connect to failed nodes.
                Defaults to 5.
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to yugabytedb db.
            table_name (str): Table name.
            schema_name (str): Schema name.
            debug (bool, optional): Debug mode. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.

        """
        from urllib.parse import urlencode

        query_params = {"load_balance": str(load_balance)}

        if topology_keys is not None:
            query_params["topology_keys"] = topology_keys
        if yb_servers_refresh_interval is not None:
            query_params["yb_servers_refresh_interval"] = yb_servers_refresh_interval
        if fallback_to_topology_keys_only:
            query_params["fallback_to_topology_keys_only"] = (
                fallback_to_topology_keys_only
            )
        if failed_host_ttl_seconds is not None:
            query_params["failed_host_ttl_seconds"] = failed_host_ttl_seconds

        query_str = urlencode(query_params)

        conn_str = (
            connection_string
            or f"yugabytedb+psycopg2://{user}:{password}@{host}:{port}/{database}?{query_str}"
        )

        session = cls._connect(conn_str, debug)
        return cls(
            session=session,
            table_name=table_name,
            schema_name=schema_name,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        schema_name: str = "public",
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> "YugabyteDBChatStore":
        """Return connection string from database parameters."""
        params = params_from_uri(uri)
        return cls.from_params(
            **params,
            table_name=table_name,
            schema_name=schema_name,
            debug=debug,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def _connect(
        cls, connection_string: str, debug: bool
    ) -> tuple[sessionmaker, sessionmaker]:
        _engine = create_engine(connection_string, echo=debug)

        return sessionmaker(_engine)

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            # Check if the specified schema exists with "CREATE" statement
            check_schema_statement = text(
                f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{self.schema_name}'"
            )
            result = session.execute(check_schema_statement).fetchone()

            # If the schema does not exist, then create it
            if not result:
                create_schema_statement = text(
                    f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"
                )
                session.execute(create_schema_statement)

            session.commit()

    def _create_tables_if_not_exists(self, base) -> None:
        with self._session() as session, session.begin():
            base.metadata.create_all(session.connection())

    def _initialize(self, base) -> None:
        self._create_schema_if_not_exists()
        self._create_tables_if_not_exists(base)

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Set messages for a key."""
        with self._session() as session:
            stmt = (
                insert(self._table_class)
                .values(
                    key=bindparam("key"), value=cast(bindparam("value"), ARRAY(JSONB))
                )
                .on_conflict_do_update(
                    index_elements=["key"],
                    set_={"value": cast(bindparam("value"), ARRAY(JSONB))},
                )
            )

            params = {
                "key": key,
                "value": [message.model_dump_json() for message in messages],
            }

            # Execute the bulk upsert
            session.execute(stmt, params)
            session.commit()

    def get_messages(self, key: str) -> list[ChatMessage]:
        """Get messages for a key."""
        with self._session() as session:
            result = session.execute(select(self._table_class).filter_by(key=key))
            result = result.scalars().first()
            if result:
                return [
                    ChatMessage.model_validate(removed_message)
                    for removed_message in result.value
                ]
            return []

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        with self._session() as session:
            stmt = (
                insert(self._table_class)
                .values(
                    key=bindparam("key"), value=cast(bindparam("value"), ARRAY(JSONB))
                )
                .on_conflict_do_update(
                    index_elements=["key"],
                    set_={"value": cast(bindparam("value"), ARRAY(JSONB))},
                )
            )
            params = {"key": key, "value": [message.model_dump_json()]}
            session.execute(stmt, params)
            session.commit()

    def delete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Delete messages for a key."""
        with self._session() as session:
            session.execute(delete(self._table_class).filter_by(key=key))
            session.commit()
        return None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        with self._session() as session:
            # First, retrieve the current list of messages
            stmt = select(self._table_class.value).where(self._table_class.key == key)
            result = session.execute(stmt).scalar_one_or_none()

            if result is None or idx < 0 or idx >= len(result):
                # If the key doesn't exist or the index is out of bounds
                return None

            # Remove the message at the given index
            removed_message = result[idx]

            stmt = text(
                f"""
                UPDATE {self._table_class.__tablename__}
                SET value = array_cat(
                               {self._table_class.__tablename__}.value[: :idx],
                               {self._table_class.__tablename__}.value[:idx+2:]
                           )
                WHERE key = :key;
                """
            )

            params = {"key": key, "idx": idx}
            session.execute(stmt, params)
            session.commit()

            return ChatMessage.model_validate(removed_message)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        with self._session() as session:
            # First, retrieve the current list of messages
            stmt = select(self._table_class.value).where(self._table_class.key == key)
            result = session.execute(stmt).scalar_one_or_none()

            if result is None or len(result) == 0:
                # If the key doesn't exist or the array is empty
                return None

            # Remove the message at the given index
            removed_message = result[-1]

            stmt = text(
                f"""
                UPDATE {self._table_class.__tablename__}
                SET value = value[1:array_length(value, 1) - 1]
                WHERE key = :key;
                """
            )
            params = {"key": key}
            session.execute(stmt, params)
            session.commit()

            return ChatMessage.model_validate(removed_message)

    def get_keys(self) -> list[str]:
        """Get all keys."""
        with self._session() as session:
            stmt = select(self._table_class.key)

            return session.execute(stmt).scalars().all()


def params_from_uri(uri: str) -> dict:
    result = urlparse(uri)
    database = result.path[1:]
    query_params = parse_qs(result.query)
    port = result.port if result.port else 5433
    return {
        "database": database,
        "user": result.username,
        "password": result.password,
        "host": result.hostname,
        "port": port,
        "load_balance": query_params.get("load_balance", ["false"])[0].lower()
        == "true",
        "topology_keys": query_params.get("topology_keys", [None])[0],
        "yb_servers_refresh_interval": int(
            query_params.get("yb_servers_refresh_interval", [300])[0]
        ),
        "fallback_to_topology_keys_only": query_params.get(
            "fallback_to_topology_keys_only", ["false"]
        )[0].lower()
        == "true",
        "failed_host_ttl_seconds": int(
            query_params.get("failed_host_ttl_seconds", [5])[0]
        ),
    }
