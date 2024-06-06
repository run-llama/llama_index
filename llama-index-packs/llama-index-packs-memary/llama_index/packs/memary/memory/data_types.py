from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class MemoryItem:
    entity: str
    date: datetime

    def __str__(self):
        return f"{self.entity}, {self.date.isoformat()}"

    def to_dict(self):
        return {'entity': self.entity, 'date': str(self.date.isoformat())}

    @classmethod
    def from_dict(cls, data):
        return cls(entity=data['entity'],
                   date=datetime.fromisoformat(data['date']))


@dataclass
class KnowledgeMemoryItem:
    entity: str
    count: int
    date: datetime

    def __str__(self):
        return f"{self.entity}, {self.count}, {self.date.isoformat()}"

    def to_dict(self):
        return {
            'entity': self.entity,
            'count': self.count,
            'date': str(self.date.isoformat())
        }

    @classmethod
    def from_dict(cls, data):
        return cls(entity=data['entity'],
                   count=data['count'],
                   date=datetime.fromisoformat(data['date']))
