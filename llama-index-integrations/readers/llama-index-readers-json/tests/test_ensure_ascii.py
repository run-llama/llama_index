import json
from pathlib import Path

import pytest

from llama_index.readers.json import JSONReader


@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("root_is_list", [False, True])
@pytest.mark.parametrize("branch", [["中文"], {"value": "中文"}])
def test_ensure_ascii_in_nested_fragments(
    tmp_path: Path, ensure_ascii: bool, root_is_list: bool, branch: object
) -> None:
    data = {"padding": "x" * 100, "branch": branch}
    file = tmp_path / "nested.json"
    file.write_text(
        json.dumps([data] if root_is_list else data, ensure_ascii=False),
        encoding="utf-8",
    )

    docs = JSONReader(
        levels_back=0, collapse_length=60, ensure_ascii=ensure_ascii
    ).load_data(str(file), extra_info={"source": "fixture"})

    assert len(docs) == 1
    assert docs[0].text.splitlines() == [
        "padding " + "x" * 100,
        "branch " + json.dumps(branch, ensure_ascii=ensure_ascii),
    ]
    assert docs[0].metadata == {"source": "fixture"}


@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("root_is_list", [False, True])
def test_ensure_ascii_controls_nested_collapse_length(
    tmp_path: Path, ensure_ascii: bool, root_is_list: bool
) -> None:
    data = {"padding": "x" * 100, "branch": ["中文"]}
    file = tmp_path / "threshold.json"
    file.write_text(
        json.dumps([data] if root_is_list else data, ensure_ascii=False),
        encoding="utf-8",
    )

    docs = JSONReader(
        levels_back=0, collapse_length=10, ensure_ascii=ensure_ascii
    ).load_data(str(file))

    # The escaped array exceeds the threshold; scalar text is not JSON-encoded.
    expected = "branch 中文" if ensure_ascii else 'branch ["中文"]'
    assert docs[0].text.splitlines()[-1] == expected


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_ensure_ascii_when_root_collapses(tmp_path: Path, ensure_ascii: bool) -> None:
    data = {"value": "中文"}
    file = tmp_path / "root.json"
    file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    docs = JSONReader(
        levels_back=0, collapse_length=60, ensure_ascii=ensure_ascii
    ).load_data(str(file))

    assert docs[0].text == json.dumps(data, ensure_ascii=ensure_ascii)


def test_ensure_ascii_in_jsonl_with_limited_ancestry(tmp_path: Path) -> None:
    data = {"padding": "x" * 100, "outer": {"padding": "y" * 100, "inner": ["中文"]}}
    file = tmp_path / "nested.jsonl"
    file.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in [data, [data]]),
        encoding="utf-8",
    )

    docs = JSONReader(
        levels_back=1, collapse_length=20, ensure_ascii=True, is_jsonl=True
    ).load_data(str(file))

    assert len(docs) == 2
    assert all(doc.text.splitlines()[-1] == 'inner ["\\u4e2d\\u6587"]' for doc in docs)
