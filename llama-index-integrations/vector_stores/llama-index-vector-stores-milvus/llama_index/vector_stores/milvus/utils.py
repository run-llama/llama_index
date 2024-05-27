from abc import ABC, abstractmethod
from typing import List, Dict
import sys
import logging

logger = logging.getLogger(__name__)


class BaseSparseEmbeddingFunction(ABC):
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        pass

    @abstractmethod
    def encode_documents(self, documents: List[str]) -> List[Dict[int, float]]:
        pass


class BGEM3SparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        except Exception as ImportError:
            error_info = (
                "Cannot import BGEM3FlagModel from FlagEmbedding. It seems it is not installed. "
                "Please install it using:\n"
                "pip install FlagEmbedding\n"
            )
            logger.fatal(error_info)
            sys.exit(1)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result


def get_defualt_sparse_embedding_function() -> BGEM3SparseEmbeddingFunction:
    return BGEM3SparseEmbeddingFunction()
