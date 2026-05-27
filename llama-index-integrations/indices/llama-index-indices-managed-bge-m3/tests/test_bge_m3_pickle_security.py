import os
import pickle

import numpy as np
import pytest

from llama_index.indices.managed.bge_m3.base import _RestrictedUnpickler


def test_restricted_unpickler_allows_expected_numpy_payload(tmp_path):
    store = {
        "dense_vecs": np.random.rand(2, 3).astype("float32"),
        "colbert_vecs": np.random.rand(2, 4, 5).astype("float32"),
        # BGEM3FlagModel returns lexical_weights as an array of dictionaries.
        "lexical_weights": np.array([{1: 0.1}, {2: 0.2}], dtype=object),
    }

    path = tmp_path / "multi_embed_store.pkl"
    with open(path, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path, "rb") as f:
        loaded = _RestrictedUnpickler(f).load()

    np.testing.assert_allclose(loaded["dense_vecs"], store["dense_vecs"])
    np.testing.assert_allclose(loaded["colbert_vecs"], store["colbert_vecs"])
    assert loaded["lexical_weights"].tolist() == store["lexical_weights"].tolist()


def test_restricted_unpickler_rejects_malicious_payload(tmp_path):
    class _Exploit:
        def __reduce__(self):
            return (os.system, ("echo exploited",))

    path = tmp_path / "multi_embed_store.pkl"
    with open(path, "wb") as f:
        pickle.dump(_Exploit(), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path, "rb") as f:
        with pytest.raises(pickle.UnpicklingError):
            _RestrictedUnpickler(f).load()

