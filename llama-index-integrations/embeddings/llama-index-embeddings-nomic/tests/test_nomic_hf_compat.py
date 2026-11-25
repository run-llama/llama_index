"""Regression coverage for the local Hugging Face Nomic embedding adapter."""

import asyncio
import unittest

import torch

from llama_index.embeddings.nomic.base import NomicHFEmbedding


class _Tokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, texts, **kwargs):
        self.seen.append(texts)
        return {
            "input_ids": torch.tensor([[1, 2]] * len(texts)),
            "attention_mask": torch.tensor([[1, 1]] * len(texts)),
        }


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return (torch.tensor([[[3.0, 4.0], [0.0, 1.0]]] * batch),)


class TestNomicHFCompatibility(unittest.TestCase):
    def test_injected_model_works_with_supported_huggingface_integration(self):
        embedding = NomicHFEmbedding(
            model_name="local-test-model",
            tokenizer_name="local-test-tokenizer",
            model=_Model(),
            tokenizer=_Tokenizer(),
            pooling="cls",
            device="cpu",
        )
        actual = embedding.get_text_embedding("example")
        self.assertEqual(len(actual), 2)
        self.assertAlmostEqual(actual[0], 0.6, places=6)
        self.assertAlmostEqual(actual[1], 0.8, places=6)

    def test_query_and_document_instructions_and_mean_pooling(self):
        tokenizer = _Tokenizer()
        embedding = NomicHFEmbedding(
            model_name="local-test-model",
            tokenizer_name="local-test-tokenizer",
            model=_Model(),
            tokenizer=tokenizer,
            pooling="mean",
            normalize=False,
            query_instruction="query:",
            text_instruction="document:",
            device="cpu",
        )
        self.assertEqual(embedding.get_query_embedding("hello"), [1.5, 2.5])
        self.assertEqual(embedding.get_text_embedding("world"), [1.5, 2.5])
        self.assertEqual(tokenizer.seen, [["query: hello"], ["document: world"]])

    def test_mean_pooling_ignores_padding_and_batches(self):
        class PaddedTokenizer(_Tokenizer):
            def __call__(self, texts, **kwargs):
                result = super().__call__(texts, **kwargs)
                result["attention_mask"][:, 1] = 0
                return result

        embedding = NomicHFEmbedding(
            model_name="local-test-model",
            model=_Model(),
            tokenizer=PaddedTokenizer(),
            pooling="mean",
            normalize=False,
            device="cpu",
        )
        self.assertEqual(
            embedding.get_text_embedding_batch(["one", "two"]),
            [[3.0, 4.0], [3.0, 4.0]],
        )

    def test_v15_embedding_uses_requested_dimension(self):
        embedding = NomicHFEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model=_Model(),
            tokenizer=_Tokenizer(),
            pooling="cls",
            dimensionality=1,
            device="cpu",
        )
        result = embedding.get_query_embedding("example")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(abs(result[0]), 1.0, places=6)

    def test_last_pooling_ignores_right_padding(self):
        class PaddedTokenizer(_Tokenizer):
            def __call__(self, texts, **kwargs):
                result = super().__call__(texts, **kwargs)
                result["attention_mask"][:, 1] = 0
                return result

        embedding = NomicHFEmbedding(
            model_name="local-test-model",
            model=_Model(),
            tokenizer=PaddedTokenizer(),
            pooling="last",
            normalize=False,
            device="cpu",
        )
        self.assertEqual(embedding.get_text_embedding("example"), [3.0, 4.0])

    def test_async_query_and_batch_match_sync(self):
        embedding = NomicHFEmbedding(
            model_name="local-test-model",
            model=_Model(),
            tokenizer=_Tokenizer(),
            normalize=False,
            device="cpu",
        )

        async def run():
            query = await embedding.aget_query_embedding("query")
            documents = await embedding.aget_text_embedding_batch(["one", "two"])
            return query, documents

        query, documents = asyncio.run(run())
        self.assertEqual(query, embedding.get_query_embedding("query"))
        self.assertEqual(documents, embedding.get_text_embedding_batch(["one", "two"]))


if __name__ == "__main__":
    unittest.main()
