from llama_index.tools.utils import create_schema_from_function


class Test:
    def test_fn(arg1: str) -> int:
        return 1


schema = create_schema_from_function("Test", Test.test_fn)
print(schema.schema())
