import re
import json
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

class DictReader(BaseReader):
    """
    A reader class that reads data from a dictionary.
    """

    def __init__(self, data={}):
        """
        Initialize the DictReader with the given data.

        Args:
            data (dict): The dictionary containing the data.
        """
        self.data = data

    def load_data(self):
        """
        Load the data from the dictionary and return a list of documents.

        Returns:
            list[Document]: A list of Document objects.
        """
        documents = list()
        json_output = json.dumps(self.data, indent=0)
        lines = json_output.split("\n")
        useful_lines = [
            line for line in lines if not re.match(r"^[{}\[\],]*$", line)
        ]
        documents.append(Document("\n".join(useful_lines)))
        return documents
