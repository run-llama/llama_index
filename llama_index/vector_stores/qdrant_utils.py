from typing import Callable, List, Tuple

import torch

SparseEncoderCallable = Callable[[List[str]], Tuple[List[List[int]], List[List[float]]]]


def default_sparse_encoder(model_id: str) -> SparseEncoderCallable:
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Could not import transformers library."
            "Please install transformers with `pip install transformers`"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")

    def compute_vectors(texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Computes vectors from logits and attention mask using ReLU, log, and max operations.
        """
        # TODO: compute sparse vectors in batches if max length is exceeded
        tokens = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            tokens = tokens.to("cuda")

        output = model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)

        # extract the vectors that are non-zero and their indices
        indices = []
        vecs = []
        for batch in tvecs:
            indices.append(batch.nonzero(as_tuple=True)[0].tolist())
            vecs.append(batch[indices[-1]].tolist())

        return indices, vecs

    return compute_vectors
