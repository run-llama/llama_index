"""JSON Reader."""

import json
import re
from typing import Any, Generator, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document


def _depth_first_yield(
    json_data: Any,
    levels_back: int,
    collapse_length: Optional[int],
    path: List[str],
    ensure_ascii: bool = False,
) -> Generator[str, None, None]:
    """Do depth first yield of all of the leaf nodes of a JSON.

    Combines keys in the JSON tree using spaces.

    If levels_back is set to 0, prints all levels.
    If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.

    """
    if isinstance(json_data, (dict, list)):
        # only try to collapse if we're not at a leaf node
        json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
        if collapse_length is not None and len(json_str) <= collapse_length:
            new_path = path[-levels_back:]
            new_path.append(json_str)
            yield " ".join(new_path)
            return
        elif isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from _depth_first_yield(
                    value, levels_back, collapse_length, new_path
                )
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from _depth_first_yield(value, levels_back, collapse_length, path)
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

        collapse_length (int): the maximum number of characters a JSON fragment
          would be collapsed in the output (levels_back needs to be not None)
          ex: if collapse_length = 10, and
          input is {a: [1, 2, 3], b: {"hello": "world", "foo": "bar"}}
          then a would be collapsed into one line, while b would not.
          Recommend starting around 100 and then adjusting from there.

    """

    def __init__(
        self,
        levels_back: Optional[int] = None,
        collapse_length: Optional[int] = None,
        ensure_ascii: bool = False,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.levels_back = levels_back
        self.collapse_length = collapse_length
        self.ensure_ascii = ensure_ascii

    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)
            if self.levels_back is None:
                # If levels_back isn't set, we just format and make each
                # line an embedding
                json_output = json.dumps(data, indent=0, ensure_ascii=self.ensure_ascii)
                lines = json_output.split("\n")
                useful_lines = [
                    line for line in lines if not re.match(r"^[{}\[\],]*$", line)
                ]
                return [Document(text="\n".join(useful_lines))]
            elif self.levels_back is not None:
                # If levels_back is set, we make the embeddings contain the labels
                # from further up the JSON tree
                lines = [
                    *_depth_first_yield(
                        data,
                        self.levels_back,
                        self.collapse_length,
                        [],
                        self.ensure_ascii,
                    )
                ]
                return [Document(text="\n".join(lines))]
            return None
