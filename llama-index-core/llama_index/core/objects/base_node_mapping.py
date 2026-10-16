"""Base object types."""

import os
import pickle
from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Sequence, TypeVar

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.utils import concat_dirs

DEFAULT_PERSIST_FNAME = "object_node_mapping.pickle"

# Classes allowed during deserialization of object node mappings.
# Restricting unpickling to this set prevents arbitrary code execution
# via crafted pickle payloads placed in the persist directory (CWE-502).
_SAFE_PICKLE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "tuple"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "complex"),
        ("builtins", "bool"),
        ("builtins", "NoneType"),
        (__name__, "SimpleObjectNodeMapping"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickler that restricts class instantiation to an allowlist.

    Prevents arbitrary code execution when loading from untrusted or
    user-configurable persist directories.
    """

    def find_class(self, module: str, name: str) -> type:
        if (module, name) in _SAFE_PICKLE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle '{module}.{name}': class not in allowlist. "
            f"If you need to load custom object types, use a purpose-built "
            f"serialization format instead of pickle."
        )


def _iter_disallowed_classes(
    obj: Any, _seen: Optional[set] = None
) -> set[tuple[str, str]]:
    """
    Collect (module, class) pairs in obj that the load allowlist rejects.

    Mirrors _RestrictedUnpickler's allowlist so persist() can fail fast
    instead of writing a file that from_persist_dir() will refuse to read.
    """
    if _seen is None:
        _seen = set()
    found: set[tuple[str, str]] = set()
    cls = type(obj)
    key = (cls.__module__, cls.__name__)
    if key not in _SAFE_PICKLE_CLASSES:
        found.add(key)
        return found
    if id(obj) in _seen:
        return found
    _seen.add(id(obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            found |= _iter_disallowed_classes(k, _seen)
            found |= _iter_disallowed_classes(v, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            found |= _iter_disallowed_classes(item, _seen)
    return found


OT = TypeVar("OT")


class BaseObjectNodeMapping(Generic[OT]):
    """Base object node mapping."""

    @classmethod
    @abstractmethod
    def from_objects(
        cls, objs: Sequence[OT], *args: Any, **kwargs: Any
    ) -> "BaseObjectNodeMapping":
        """
        Initialize node mapping from a list of objects.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """

    def validate_object(self, obj: OT) -> None:
        """Validate object."""

    def add_object(self, obj: OT) -> None:
        """
        Add object.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """
        self.validate_object(obj)
        self._add_object(obj)

    @property
    @abstractmethod
    def obj_node_mapping(self) -> Dict[Any, Any]:
        """The mapping data structure between node and object."""

    @abstractmethod
    def _add_object(self, obj: OT) -> None:
        """
        Add object.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """

    @abstractmethod
    def to_node(self, obj: OT) -> BaseNode:
        """To node."""

    def to_nodes(self, objs: Sequence[OT]) -> Sequence[BaseNode]:
        return [self.to_node(obj) for obj in objs]

    def from_node(self, node: BaseNode) -> OT:
        """From node."""
        obj = self._from_node(node)
        self.validate_object(obj)
        return obj

    @abstractmethod
    def _from_node(self, node: BaseNode) -> OT:
        """From node."""

    @abstractmethod
    def persist(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> None:
        """Persist objs."""

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> "BaseObjectNodeMapping[OT]":
        """Load from serialization."""
        obj_node_mapping = None
        errors = []
        for cls in BaseObjectNodeMapping.__subclasses__():  # type: ignore
            try:
                obj_node_mapping = cls.from_persist_dir(
                    persist_dir=persist_dir,
                    obj_node_mapping_fname=obj_node_mapping_fname,
                )
                break
            except (NotImplementedError, pickle.PickleError) as err:
                # raise unhandled exception otherwise
                errors.append(err)
        if obj_node_mapping:
            return obj_node_mapping
        else:
            raise Exception(errors)


class SimpleObjectNodeMapping(BaseObjectNodeMapping[Any]):
    """
    General node mapping that works for any obj.

    More specifically, any object with a meaningful string representation.

    """

    def __init__(self, objs: Optional[Sequence[Any]] = None) -> None:
        objs = objs or []
        for obj in objs:
            self.validate_object(obj)
        self._objs = {hash(str(obj)): obj for obj in objs}

    @classmethod
    def from_objects(
        cls, objs: Sequence[Any], *args: Any, **kwargs: Any
    ) -> "SimpleObjectNodeMapping":
        return cls(objs)

    @property
    def obj_node_mapping(self) -> Dict[int, Any]:
        return self._objs

    @obj_node_mapping.setter
    def obj_node_mapping(self, mapping: Dict[int, Any]) -> None:
        self._objs = mapping

    def _add_object(self, obj: Any) -> None:
        self._objs[hash(str(obj))] = obj

    def to_node(self, obj: Any) -> TextNode:
        return TextNode(id_=str(hash(str(obj))), text=str(obj))

    def _from_node(self, node: BaseNode) -> Any:
        return self._objs[hash(node.get_content(metadata_mode=MetadataMode.NONE))]

    def persist(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> None:
        """
        Persist object node mapping.

        NOTE: This may fail depending on whether the object types are
        pickle-able. Only objects whose classes are in the load allowlist
        (_SAFE_PICKLE_CLASSES) can round-trip: from_persist_dir() refuses
        everything else, so persist() raises instead of writing a file it
        could never read back.
        """
        disallowed: set[tuple[str, str]] = set()
        for obj in self._objs.values():
            disallowed |= _iter_disallowed_classes(obj)
        if disallowed:
            names = ", ".join(sorted(f"{mod}.{name}" for mod, name in disallowed))
            raise ValueError(
                f"Cannot persist objects of type {names}: from_persist_dir() "
                "only loads allowlisted classes, so the persisted file would "
                "be unreadable. Convert custom objects to builtin types "
                "before persisting."
            )
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        obj_node_mapping_path = concat_dirs(persist_dir, obj_node_mapping_fname)
        try:
            with open(obj_node_mapping_path, "wb") as f:
                pickle.dump(self, f)
        except pickle.PickleError as err:
            raise ValueError("Objs is not pickleable") from err

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> "SimpleObjectNodeMapping":
        obj_node_mapping_path = concat_dirs(persist_dir, obj_node_mapping_fname)
        try:
            with open(obj_node_mapping_path, "rb") as f:
                simple_object_node_mapping = _RestrictedUnpickler(f).load()
        except pickle.PickleError as err:
            raise ValueError("Objs cannot be loaded.") from err
        return simple_object_node_mapping
