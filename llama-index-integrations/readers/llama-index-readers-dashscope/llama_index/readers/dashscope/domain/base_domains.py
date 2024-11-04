from abc import ABC


class DictToObject(ABC):
    @classmethod
    def from_dict(cls, data: dict):
        pass
