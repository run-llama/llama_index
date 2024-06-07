import json
import logging

from llama_index.packs.memary.memory import BaseMemory
from llama_index.packs.memary.memory.data_types import KnowledgeMemoryItem, MemoryItem


class EntityKnowledgeStore(BaseMemory):
    def __len__(self) -> int:
        """Returns the number of items in the memory."""
        return len(self.knowledge_memory)

    def init_memory(self):
        """Initializes memory.

        self.entity_memory: list[EntityMemoryItem]
        """
        self.load_memory_from_file()
        if self.entity:
            self.add_memory(self.entity)

    @property
    def return_memory(self):
        return self.knowledge_memory

    def load_memory_from_file(self):
        try:
            with open(self.file_name) as file:
                self.knowledge_memory = [
                    KnowledgeMemoryItem.from_dict(item) for item in json.load(file)
                ]
            logging.info(
                f"Entity Knowledge Memory loaded from {self.file_name} successfully."
            )
        except FileNotFoundError:
            logging.info(
                "File not found. Starting with an empty entity knowledge memory."
            )

    def add_memory(self, memory_stream: list[MemoryItem]):
        """Add new memory to the entity knowledge store.

        We should convert the memory to knowledge memory and then update the knowledge memory.

        Args:
            memory_stream (list): list of MemoryItem
        """
        knowledge_meory = self._convert_memory_to_knowledge_memory(memory_stream)
        self._update_knowledge_memory(knowledge_meory)

    def _update_knowledge_memory(self, knowledge_memory: list):
        """Update self.knowledge memory with new knowledge memory items.

        Args:
            knowledge_memory (list): list of KnowledgeMemoryItem
        """
        for item in knowledge_memory:
            for i, entity in enumerate(self.knowledge_memory):
                if entity.entity == item.entity:
                    self.knowledge_memory[i].date = item.date
                    self.knowledge_memory[i].count += item.count
                    break
            else:
                self.knowledge_memory.append(item)

    def _convert_memory_to_knowledge_memory(
        self, memory_stream: list
    ) -> list[KnowledgeMemoryItem]:
        """Converts memory to knowledge memory.

        Returns:
            knowledge_memory (list): list of KnowledgeMemoryItem
        """
        knowledge_memory = []

        entities = {item.entity for item in memory_stream}
        for entity in entities:
            memory_dates = [
                item.date for item in memory_stream if item.entity == entity
            ]
            knowledge_memory.append(
                KnowledgeMemoryItem(entity, len(memory_dates), max(memory_dates))
            )
        return knowledge_memory

    def get_memory(self) -> list[KnowledgeMemoryItem]:
        return self.knowledge_memory
