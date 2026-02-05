from llama_index.core.llms.utils import parse_partial_json
import json

def test_parse_partial_json_repro():
    cases = [
        ('{"name": "Jack', {"name": "Jack"}),
        ('{"a": "foo", "b": "bar', {"a": "foo", "b": "bar"}),
        ('{"a": "foo", "b', {"a": "foo"}), # Incomplete key should probably be dropped or handle differently
    ]
    
    for s, expected in cases:
        try:
            result = parse_partial_json(s)
            print(f"Input: {s}")
            print(f"Expected: {expected}")
            print(f"Result:   {result}")
            if result == expected:
                print("✅ PASS")
            else:
                print("❌ FAIL")
        except Exception as e:
            print(f"Input: {s}")
            print(f"Error: {e}")
            print("❌ FAIL")
        print("-" * 20)

if __name__ == "__main__":
    test_parse_partial_json_repro()
