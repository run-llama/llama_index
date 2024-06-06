import json
import logging

from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class BaseMemory(ABC):

    def __init__(self, file_name: str, entity: str = None):
        """Initializes the memory storage."""
        self.file_name = file_name
        self.entity = entity
        self.memory = []
        self.knowledge_memory = []
        self.init_memory()

    @abstractmethod
    def __len__(self):
        """Returns the number of items in the memory."""
        pass

    @abstractmethod
    def init_memory(self):
        """Initializes memory."""
        pass

    @abstractmethod
    def load_memory_from_file(self):
        """Loads memory from a file."""
        pass

    @abstractmethod
    def add_memory(self, data):
        """Adds new memory data."""
        pass

    @abstractmethod
    def get_memory(self):
        pass

    @property
    def return_memory(self):
        return self.memory

    def remove_old_memory(self, days):
        """Removes memory items older than a specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        self.memory = [
            item for item in self.return_memory if item.date >= cutoff_date
        ]
        logging.info("Old memory removed successfully.")

    def save_memory(self):
        if self.file_name:
            with open(self.file_name, 'w') as file:
                json.dump([item.to_dict() for item in self.return_memory],
                          file,
                          default=str,
                          indent=4)
                logging.info(f"Memory saved to {self.file_name} successfully.")
        else:
            logging.info("No file name provided. Memory not saved.")

    def get_memory_by_index(self, index):
        if 0 <= index < len(self.memory):
            return self.memory[index]
        else:
            return None

    def remove_memory_by_index(self, index):
        if 0 <= index < len(self.memory):
            del self.memory[index]
            logging.info("Memory item removed successfully.")
            return True
        else:
            logging.info("Invalid index. Memory item not removed.")
            return False

    def clear_memory(self):
        self.memory = []
        if self.file_name:
            with open(self.file_name, 'w') as file:
                json.dump([], file, indent=4)
                logging.info(
                    f"Memory cleared and saved to {self.file_name} successfully."
                )
        else:
            logging.info("No file name provided. Memory not cleared or saved.")
