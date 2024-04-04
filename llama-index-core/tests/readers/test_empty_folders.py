import unittest
import os
import tempfile
from unittest.mock import MagicMock
from llama_index.core.readers.file.base import SimpleDirectoryReader


class TestSimpleDirectoryReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after testing
        os.rmdir(self.temp_dir)

    def test_load_data_returns_empty_list_when_input_files_is_none(self):
        result = SimpleDirectoryReader(
            self.temp_dir, file_extractor=MagicMock(), file_metadata=MagicMock()
        ).load_data()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
