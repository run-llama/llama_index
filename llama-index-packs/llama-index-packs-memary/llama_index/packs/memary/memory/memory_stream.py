import json
import logging
from datetime import datetime

from llama_index.packs.memary.memory import BaseMemory
from llama_index.packs.memary.memory.data_types import MemoryItem

logging.basicConfig(level=logging.INFO)


class MemoryStream(BaseMemory):
    def __len__(self) -> int:
        """Returns the number of items in the memory."""
        return len(self.memory)

    def init_memory(self):
        """Initializes memory.

        self.memory: list[MemoryItem]
        """
        self.load_memory_from_file()
        if self.entity:
            self.add_memory(self.entity)

    @property
    def return_memory(self):
        return self.memory

    def add_memory(self, entities):
        self.memory.extend(
            [
                MemoryItem(str(entity), datetime.now().replace(microsecond=0))
                for entity in entities
            ]
        )

    def get_memory(self) -> list[MemoryItem]:
        return self.memory

    def load_memory_from_file(self):
        try:
            with open(self.file_name) as file:
                self.memory = [MemoryItem.from_dict(item) for item in json.load(file)]
            logging.info(f"Memory loaded from {self.file_name} successfully.")
        except FileNotFoundError:
            logging.info("File not found. Starting with an empty memory.")
