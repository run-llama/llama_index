from typing import List

import numpy as np
import pandas as pd
from llama_index.vector_stores.lancedb import _to_llama_similarities

data_stub = {
    "id": [1, 2, 3],
    "doc_id": ["doc1", "doc2", "doc3"],
    "vector": [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])],
    "text": ["text1", "text2", "text3"],
    "file_name": ["file1.txt", "file2.txt", "file3.txt"],
    "_node_content": ["content1", "content2", "content3"],
    "document_id": ["doc_id1", "doc_id2", "doc_id3"],
    "ref_doc_id": ["ref1", "ref2", "ref3"],
}


def test_to_llama_similarities_from_df_w_score() -> None:
    data = dict(data_stub)
    scores: List[float] = [9, 9 - np.log(2), 9 - np.log(4)]

    # lance provides 'score' in reverse natural sort test should as well
    reversed_sort = scores.copy()
    reversed_sort.sort(reverse=True)
    assert np.array_equal(reversed_sort, scores)  # gut check setup

    data["score"] = scores
    df = pd.DataFrame(data)
    llama_sim_array = _to_llama_similarities(df)
    assert np.allclose(llama_sim_array, [1, 0.5, 0.25])


def test_to_llama_similarities_from_df_w_distance() -> None:
    data = dict(data_stub)
    distances: List[float] = [np.log(4 / 3), np.log(2), np.log(4)]

    # lance provides '_distance' by natural sort test should as well
    natural_sort = distances.copy()
    natural_sort.sort()
    assert np.array_equal(natural_sort, distances)  # gut check setup

    data["_distance"] = distances
    df = pd.DataFrame(data)
    llama_sim_array = _to_llama_similarities(df)
    assert np.allclose(llama_sim_array, [0.75, 0.5, 0.25])


def test_to_llama_similarity_from_df_ordinal() -> None:
    data = dict(data_stub)
    df = pd.DataFrame(data)
    llama_sim_array = _to_llama_similarities(df)
    assert np.allclose(llama_sim_array, [1, 0.5, 0])
