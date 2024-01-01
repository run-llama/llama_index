import json
import logging
import sys
from typing import Any, Callable, List, Optional
from urllib.parse import urlparse

from redis import Redis, RedisCluster

from llama_index.bridge.pydantic import Field, root_validator
from llama_index.llms.llm import LLM
from llama_index.llms.types import ChatMessage, MessageRole
from llama_index.memory.types import BaseMemory
from llama_index.utils import get_tokenizer

DEFUALT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


# Convert a ChatMessage to a json object for Redis
def _message_to_dict(message: ChatMessage) -> dict:
    return {"type": message.role, "content": message.content}


# Convert the json object in Redis to a ChatMessage
def _dict_to_message(d: dict) -> ChatMessage:
    return ChatMessage(role=d["type"], content=d["content"])


def _get_client(redis_url: str, **kwargs: Any) -> Any:
    """Get a redis client from the connection url given. This helper accepts
    urls for Redis server (TCP with/without TLS or UnixSocket) as well as
    Redis Sentinel connections.

    Redis Cluster is not supported.

    Before creating a connection the existence of the database driver is checked
    an and ValueError raised otherwise

    To use, you should have the ``redis`` python package installed.

    Example:
        .. code-block:: python

            redis_client = get_client(
                redis_url="redis://username:password@localhost:6379"
            )

    To use a redis replication setup with multiple redis server and redis sentinels
    set "redis_url" to "redis+sentinel://" scheme. With this url format a path is
    needed holding the name of the redis service within the sentinels to get the
    correct redis server connection. The default service name is "mymaster". The
    optional second part of the path is the redis db number to connect to.

    An optional username or password is used for booth connections to the rediserver
    and the sentinel, different passwords for server and sentinel are not supported.
    And as another constraint only one sentinel instance can be given:

    Example:
        .. code-block:: python

            redis_client = get_client(
                redis_url="redis+sentinel://username:password@sentinelhost:26379/mymaster/0"
            )
    """
    # Initialize with necessary components.
    try:
        import redis
    except ImportError:
        raise ImportError(
            "Could not import redis python package. "
            "Please install it with `pip install redis>=4.1.0`."
        )

    redis_client: Any
    # check if normal redis:// or redis+sentinel:// url
    if redis_url.startswith("redis+sentinel"):
        redis_client = _redis_sentinel_client(redis_url, **kwargs)
    elif redis_url.startswith("rediss+sentinel"):  # sentinel with TLS support enables
        kwargs["ssl"] = True
        if "ssl_cert_reqs" not in kwargs:
            kwargs["ssl_cert_reqs"] = "none"
        redis_client = _redis_sentinel_client(redis_url, **kwargs)
    else:
        # connect to redis server from url, reconnect with cluster client if needed
        redis_client = redis.from_url(redis_url, **kwargs)
        if _check_for_cluster(redis_client):
            redis_client.close()
            redis_client = _redis_cluster_client(redis_url, **kwargs)
    return redis_client


def _redis_sentinel_client(redis_url: str, **kwargs: Any) -> Redis:
    """Helper method to parse an (un-official) redis+sentinel url
    and create a Sentinel connection to fetch the final redis client
    connection to a replica-master for read-write operations.

    If username and/or password for authentication is given the
    same credentials are used for the Redis Sentinel as well as Redis Server.
    With this implementation using a redis url only it is not possible
    to use different data for authentication on booth systems.
    """
    import redis

    parsed_url = urlparse(redis_url)
    # sentinel needs list with (host, port) tuple, use default port if none available
    sentinel_list = [(parsed_url.hostname or "localhost", parsed_url.port or 26379)]
    if parsed_url.path:
        # "/mymaster/0" first part is service name, optional second part is db number
        path_parts = parsed_url.path.split("/")
        service_name = path_parts[1] or "mymaster"
        if len(path_parts) > 2:
            kwargs["db"] = path_parts[2]
    else:
        service_name = "mymaster"

    sentinel_args = {}
    if parsed_url.password:
        sentinel_args["password"] = parsed_url.password
        kwargs["password"] = parsed_url.password
    if parsed_url.username:
        sentinel_args["username"] = parsed_url.username
        kwargs["username"] = parsed_url.username

    # check for all SSL related properties and copy them into sentinel_kwargs too,
    # add client_name also
    for arg in kwargs:
        if arg.startswith("ssl") or arg == "client_name":
            sentinel_args[arg] = kwargs[arg]

    # sentinel user/pass is part of sentinel_kwargs, user/pass for redis server
    # connection as direct parameter in kwargs
    sentinel_client = redis.sentinel.Sentinel(
        sentinel_list, sentinel_kwargs=sentinel_args, **kwargs
    )

    # redis server might have password but not sentinel - fetch this error and try
    # again without pass, everything else cannot be handled here -> user needed
    try:
        sentinel_client.execute_command("ping")
    except redis.exceptions.AuthenticationError:
        exception_info = sys.exc_info()
        exception = exception_info[1] or None
        if exception is not None and "no password is set" in exception.args[0]:
            logging.warning(
                msg="Redis sentinel connection configured with password but Sentinel \
answered NO PASSWORD NEEDED - Please check Sentinel configuration"
            )
            sentinel_client = redis.sentinel.Sentinel(sentinel_list, **kwargs)
        else:
            raise

    return sentinel_client.master_for(service_name)


def _check_for_cluster(redis_client: Redis) -> bool:
    import redis

    try:
        cluster_info = redis_client.info("cluster")
        return cluster_info["cluster_enabled"] == 1
    except redis.exceptions.RedisError:
        return False


def _redis_cluster_client(redis_url: str, **kwargs: Any) -> RedisCluster:
    from redis.cluster import RedisCluster

    return RedisCluster.from_url(redis_url, **kwargs)


class RedisChatMemoryBuffer(BaseMemory):
    """Simple buffer for storing chat history."""

    session_id: str
    key_prefix: str
    token_limit: int
    tokenizer_fn: Callable[[str], List] = Field(
        # NOTE: mypy does not handle the typing here well, hence the cast
        default_factory=get_tokenizer,
        exclude=True,
    )
    chat_history: List[ChatMessage] = Field(default_factory=list)
    redis_url: str
    redis_client: Any
    ttl: Optional[int]

    @root_validator(pre=True)
    def validate_redis(cls, values: dict) -> dict:
        # Validate token limit
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
            raise ValueError("Token limit must be set and greater than 0.")

        # Validate tokenizer -- this avoids errors when loading from json/dict
        tokenizer_fn = values.get("tokenizer_fn", None)
        if tokenizer_fn is None:
            values["tokenizer_fn"] = get_tokenizer()

        return values

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
        **kwargs: Any
    ) -> "RedisChatMemoryBuffer":
        """Create a chat memory buffer from an LLM."""
        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(context_window * DEFUALT_TOKEN_LIMIT_RATIO)
        elif token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT

        try:
            import redis
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        session_id = kwargs.get("session_id", None)
        redis_url = kwargs.get("redis_url", "redis://localhost:6379/0")
        key_prefix = kwargs.get("key_prefix", "chat_history:")
        ttl = kwargs.get("ttl", None)

        try:
            redis_client = _get_client(redis_url=redis_url)
        except redis.exceptions.ConnectionError as error:
            raise ValueError("Could not connect to redis database.")

        instance = cls(
            session_id=session_id,
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            chat_history=chat_history or [],
            redis_url=redis_url,
            redis_client=redis_client,
            key_prefix=key_prefix,
            ttl=ttl,
        )

        if chat_history is not None and len(chat_history) > 0:
            instance.set(chat_history)

        return instance

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    def get(self, initial_token_count: int = 0, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        chat_history = self.get_all()

        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        message_count = len(chat_history)
        token_count = (
            self._token_count_for_message_count(message_count) + initial_token_count
        )

        while token_count > self.token_limit and message_count > 1:
            message_count -= 1
            if chat_history[-message_count].role == MessageRole.ASSISTANT:
                # we cannot have an assistant message at the start of the chat history
                # if after removal of the first, we have an assistant message,
                # we need to remove the assistant message too
                message_count -= 1

            token_count = (
                self._token_count_for_message_count(message_count) + initial_token_count
            )

        # catch one message longer than token limit
        if token_count > self.token_limit or message_count <= 0:
            return []

        return chat_history[-message_count:]

    @property
    def key(self) -> str:
        """Construct the record key to use."""
        return self.key_prefix + self.session_id

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        items = self.redis_client.lrange(self.key, 0, -1)
        items_json = [json.loads(m.decode("utf-8")) for m in items[::-1]]
        return [_dict_to_message(d) for d in items_json]

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        item = json.dumps(_message_to_dict(message))
        self.redis_client.lpush(self.key, item)
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)
        # self.chat_history.append(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.redis_client.delete(self.key)
        for message in messages:
            self.put(message)

        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def reset(self) -> None:
        """Reset chat history."""
        return self.redis_client.delete(self.key)

    def _token_count_for_message_count(self, message_count: int) -> int:
        chat_history = self.get_all()
        if message_count <= 0:
            return 0
        msg_str = " ".join(str(m.content) for m in chat_history[-message_count:])
        return len(self.tokenizer_fn(msg_str))
