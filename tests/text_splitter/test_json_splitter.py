from llama_index.text_splitter.json_splitter import JSONSplitter


def test_split_empty_text() -> None:
    json_splitter = JSONSplitter()
    input_text = ""
    result = json_splitter.split_text(input_text)
    assert result == []


def test_split_valid_json() -> None:
    json_splitter = JSONSplitter(levels_back=0)
    input_text = '[{"name": "John", "age": 30}, {"name": "Alice", "age": 25}]'
    result = json_splitter.split_text(input_text)
    assert len(result) == 2
    assert "name John\nage 30" == result[0]
    assert "name Alice\nage 25" == result[1]


def test_split_valid_json_defaults() -> None:
    json_splitter = JSONSplitter()
    input_text = '[{"name": "John", "age": 30}]'
    result = json_splitter.split_text(input_text)
    assert len(result) == 1
    assert "name John\nage 30" == result[0]


def test_split_valid_dict_json() -> None:
    json_splitter = JSONSplitter(levels_back=0)
    input_text = '{"name": "John", "age": 30}'
    result = json_splitter.split_text(input_text)
    assert len(result) == 1
    assert "name John\nage 30" == result[0]


def test_split_invalid_json() -> None:
    json_splitter = JSONSplitter()
    input_text = '{"name": "John", "age": 30,}'
    result = json_splitter.split_text(input_text)
    assert result == []


def test_split_with_levels_back() -> None:
    input_text = '[{"a": {"b": {"c": "value", "d": "value2"}}}]'
    json_splitter = JSONSplitter(levels_back=2)
    result = json_splitter.split_text(input_text)
    assert len(result) == 1
    assert "b c value\nb d value2" == result[0]


def test_split_with_jq_path() -> None:
    input_text = '[{"a": {"b": {"c": "value", "d": "value2"}}}]'
    json_splitter = JSONSplitter(jq_path=".[].a.b")
    result = json_splitter.split_text(input_text)
    assert len(result) == 1
    assert "c value\nd value2" == result[0]


def test_split_with_jq_path_and_levels() -> None:
    input_text = (
        '[{"a": {"b": {"c": {"d": "value"}}}, "1": {"2": {"3": {"4": "value"}}}}]'
    )
    json_splitter = JSONSplitter(levels_back=1, jq_path=".[].a.b")
    result = json_splitter.split_text(input_text)
    print(result)
    assert len(result) == 1
    assert "d value" == result[0]
