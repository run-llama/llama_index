import unittest
from unittest.mock import MagicMock
from llama_index.readers.google import GoogleDriveReader


class TestGoogleDriveReader(unittest.TestCase):
    def test_load_data_with_drive_id(self):
        # Mock the necessary objects and methods
        mock_credentials = MagicMock()
        mock_drive = MagicMock()
        reader = GoogleDriveReader()
        reader._get_credentials = MagicMock(return_value=(mock_credentials, mock_drive))
        reader._load_from_folder = MagicMock(return_value=["document1", "document2"])

        # Test with a specific drive_id
        drive_id = "example_drive_id"
        folder_id = "example_folder_id"
        result = reader.load_data(drive_id=drive_id, folder_id=folder_id)

        # Assert that the correct methods are called and the correct result is returned
        reader._get_credentials.assert_called_once()
        reader._load_from_folder.assert_called_once_with(
            drive_id, folder_id, None, None
        )
        self.assertEqual(result, ["document1", "document2"])


if __name__ == "__main__":
    unittest.main()
