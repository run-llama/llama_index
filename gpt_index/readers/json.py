"""JSON Reader."""

import json
import re
from typing import Dict, Generator, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


def _depth_first_yield(
    json_data: Dict, levels_back: int, path: List[str]
) -> Generator[str, None, None]:
    """Do depth first yield of all of the leaf nodes of a JSON.

    Combines keys in the JSON tree using spaces.

    If levels_back is set to 0, prints all levels.

    """
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            new_path = path[:]
            new_path.append(key)
            yield from _depth_first_yield(value, levels_back, new_path)
    elif isinstance(json_data, list):
        for _, value in enumerate(json_data):
            yield from _depth_first_yield(value, levels_back, path)
    else:
        new_path = path[-levels_back:]
        new_path.append(str(json_data))
        yield " ".join(new_path)


class JSONReader(BaseReader):
    """JSON reader.

    Reads JSON documents with options to help suss out relationships between nodes.

    Args:
        levels_back (int): the number of levels to go back in the JSON tree, 0
        if you want all levels. If levels_back is None, then we just format the
        JSON and make each line an embedding

    """

    def __init__(self, levels_back: Optional[int] = None) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.levels_back = levels_back

    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        with open(input_file, "r") as f:
            data = json.load(f)
            if self.levels_back is None:
                # If levels_back isn't set, we just format and make each
                # line an embedding
                json_output = json.dumps(data, indent=0)
                lines = json_output.split("\n")
                useful_lines = [
                    line for line in lines if not re.match(r"^[{}\[\],]*$", line)
                ]
                return [Document("\n".join(useful_lines))]
            elif self.levels_back is not None:
                # If levels_back is set, we make the embeddings contain the labels
                # from further up the JSON tree
                lines = [*_depth_first_yield(data, self.levels_back, [])]
                return [Document("\n".join(lines))]
