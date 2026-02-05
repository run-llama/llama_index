import llama_index.core.llms.utils as utils
print(f"Imported from: {utils.__file__}")
from llama_index.core.llms.utils import parse_partial_json
import json

def test_parse_partial_json_repro():
    cases = [
        ('{"name": "Jack', {"name": "Jack"}),
    ]
    
    for s, expected in cases:
        result = parse_partial_json(s)
        print(f"Result: {result}")

if __name__ == "__main__":
    test_parse_partial_json_repro()
