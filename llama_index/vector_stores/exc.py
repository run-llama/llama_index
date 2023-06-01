__all__ = [
    "VecsException",
    "CollectionAlreadyExists",
    "CollectionNotFound",
    "ArgError",
    "FilterError",
    "IndexNotFound",
    "Unreachable",
]


class VecsException(Exception):
    ...


class CollectionAlreadyExists(VecsException):
    ...


class CollectionNotFound(VecsException):
    ...


class ArgError(VecsException):
    ...


class FilterError(VecsException):
    ...


class IndexNotFound(VecsException):
    ...


class Unreachable(VecsException):
    ...
