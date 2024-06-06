import os
import unittest
from datetime import datetime, timedelta
from llama_index.packs.memary import ChatAgent, EntityKnowledgeStore, MemoryStream
from llama_index.packs.memary.memory.data_types import KnowledgeMemoryItem, MemoryItem


class TestEntityKnowledgeStore(unittest.TestCase):

    def setUp(self):
        self.file_name = "tests/memory/test_knowledge_memory.json"
        self.entity_knowledge_store = EntityKnowledgeStore(
            file_name=self.file_name)

    def tearDown(self):
        # Clean up test file after each test
        try:
            os.remove(self.file_name)
        except FileNotFoundError:
            pass

    def test_add_memory(self):
        data = [
            MemoryItem("test_entity",
                       datetime.now().replace(microsecond=0))
        ]
        self.entity_knowledge_store.add_memory(data)
        assert len(self.entity_knowledge_store.knowledge_memory) == 1
        assert isinstance(self.entity_knowledge_store.knowledge_memory[0],
                          KnowledgeMemoryItem)

    def test_convert_memory_to_knowledge_memory(self):
        data = [
            MemoryItem("test_entity",
                       datetime.now().replace(microsecond=0))
        ]
        converted_data = self.entity_knowledge_store._convert_memory_to_knowledge_memory(
            data)
        assert len(converted_data) == 1
        assert isinstance(converted_data[0], KnowledgeMemoryItem)

    def test_update_knowledge_memory(self):
        data = [
            KnowledgeMemoryItem("knowledge_entity", 1,
                                datetime.now().replace(microsecond=0))
        ]
        self.entity_knowledge_store._update_knowledge_memory(data)
        assert len(self.entity_knowledge_store.knowledge_memory) == 1
        assert self.entity_knowledge_store.knowledge_memory[0] == data[0]


class TestMemoryStream(unittest.TestCase):

    def setUp(self):
        self.file_name = "tests/test_memory.json"
        self.memory_stream = MemoryStream(file_name=self.file_name)

    def tearDown(self):
        # Clean up test file after each test
        try:
            os.remove(self.file_name)
        except FileNotFoundError:
            pass

    def test_save_and_load_memory(self):
        data = [
            MemoryItem("test_entity",
                       datetime.now().replace(microsecond=0))
        ]
        self.memory_stream.add_memory(data)
        self.memory_stream.save_memory()
        new_memory_stream = MemoryStream(file_name=self.file_name)
        self.assertEqual(len(new_memory_stream), len(self.memory_stream))
        self.assertEqual(new_memory_stream.get_memory(),
                         self.memory_stream.get_memory())
