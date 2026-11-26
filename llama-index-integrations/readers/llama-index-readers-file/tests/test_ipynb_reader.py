import json

from llama_index.readers.file.ipynb.base import IPYNBReader


def _write_notebook(path):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["print('hello')"],
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def test_load_data_accepts_str_path(tmp_path) -> None:
    # Used to raise: AttributeError: 'str' object has no attribute 'name'
    notebook = _write_notebook(tmp_path / "sample.ipynb")
    reader = IPYNBReader()
    assert isinstance(reader.load_data(str(notebook)), list)


def test_load_data_accepts_path_object(tmp_path) -> None:
    notebook = _write_notebook(tmp_path / "sample.ipynb")
    reader = IPYNBReader()
    assert isinstance(reader.load_data(notebook), list)


def test_str_and_path_inputs_produce_same_output(tmp_path) -> None:
    notebook = _write_notebook(tmp_path / "sample.ipynb")
    reader = IPYNBReader()
    assert reader.load_data(str(notebook)) == reader.load_data(notebook)
