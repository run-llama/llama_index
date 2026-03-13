"""Utility functions for sparse vector encoding in EndeeVectorStore.

This module provides factory functions for initializing sparse encoders used in
hybrid search, supporting both FastEmbed and Transformers backends.
"""

import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Mapping of model aliases to HuggingFace model identifiers
SUPPORTED_SPARSE_MODELS = {
    "splade_pp": "prithivida/Splade_PP_en_v1",
}





def _initialize_sparse_encoder_fastembed(
    model_name: str,
    batch_size: int = 256,
    cache_dir: Optional[str] = None,
    threads: Optional[int] = None,
) -> Callable:
    """Initialize a sparse encoder using FastEmbed (recommended for production).

    FastEmbed provides optimized ONNX-based inference for sparse models like SPLADE,
    offering better performance than pure PyTorch implementations.

    Args:
        model_name: Model name or alias (e.g., "splade_pp"). Will be resolved to
            HuggingFace model ID via SUPPORTED_SPARSE_MODELS.
        batch_size: Number of texts to encode in each batch (default: 256).
        cache_dir: Optional directory for caching downloaded models.
        threads: Number of threads for ONNX inference. If None, uses default.

    Returns:
        Callable: Function that takes List[str] and returns Tuple[List[List[int]],
            List[List[float]]] containing sparse indices and values.

    Raises:
        ImportError: If fastembed package is not installed.
    """
    try:
        from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
    except ImportError as e:
        raise ImportError(
            "FastEmbed is required for hybrid search but not installed.\n"
            "Installation options:\n"
            "  • CPU:  pip install endee-llamaindex[hybrid]\n"
            "  • GPU:  pip install endee-llamaindex[hybrid-gpu]\n"
            "  • Or:   pip install fastembed\n"
            "For dense-only search, create vector store without sparse_dim or model_name."
        ) from e

    # Resolve model alias to HuggingFace model ID
    resolved_model_name = SUPPORTED_SPARSE_MODELS.get(model_name, model_name)
    logger.debug(f"Resolved model name '{model_name}' -> '{resolved_model_name}'")

    # Try GPU initialization first, fall back to CPU if CUDA unavailable
    try:
        model = SparseTextEmbedding(
            resolved_model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=["CUDAExecutionProvider"],
        )
        logger.info(f"Initialized sparse encoder '{resolved_model_name}' with GPU acceleration")
    except Exception as gpu_error:
        logger.debug(f"GPU initialization failed ({gpu_error}), falling back to CPU")
        model = SparseTextEmbedding(
            resolved_model_name,
            cache_dir=cache_dir,
            threads=threads
        )
        logger.info(f"Initialized sparse encoder '{resolved_model_name}' with CPU inference")

    def compute_vectors(texts: List[str]):
        """Compute sparse vectors (indices, values) for a batch of texts.

        Args:
            texts: List of text strings to encode.

        Returns:
            Tuple containing:
                - indices: List of lists of non-zero indices
                - values: List of lists of corresponding values
        """
        embeddings = model.embed(texts, batch_size=batch_size)
        indices: List[List[int]] = []
        values: List[List[float]] = []
        for embedding in embeddings:
            # Convert numpy arrays to Python lists for JSON serialization
            indices.append(embedding.indices.tolist())
            values.append(embedding.values.tolist())
        return indices, values

    return compute_vectors


def _initialize_sparse_encoder_transformers(model_name: str) -> Callable:
    """Initialize a sparse encoder using HuggingFace Transformers.

    This provides a fallback implementation using PyTorch and Transformers when
    FastEmbed is not available. Performance is typically slower than FastEmbed's
    ONNX-based implementation.

    Args:
        model_name: Model name or alias (e.g., "splade_pp"). Will be resolved to
            HuggingFace model ID via SUPPORTED_SPARSE_MODELS.

    Returns:
        Callable: Function that takes List[str] and returns Tuple[List[List[int]],
            List[List[float]]] containing sparse indices and values.

    Raises:
        ImportError: If transformers or torch packages are not installed.
    """
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Transformers and PyTorch are required but not installed.\n"
            'Please install with: pip install "transformers[torch]"'
        ) from e

    # Resolve model alias to HuggingFace model ID
    resolved_model_name = SUPPORTED_SPARSE_MODELS.get(model_name, model_name)
    logger.debug(f"Resolved model name '{model_name}' -> '{resolved_model_name}'")

    # Load tokenizer and model from HuggingFace
    logger.debug(f"Loading tokenizer and model from HuggingFace: {resolved_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    model = AutoModelForMaskedLM.from_pretrained(resolved_model_name)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info(f"Initialized sparse encoder '{resolved_model_name}' with GPU acceleration")
    else:
        logger.info(f"Initialized sparse encoder '{resolved_model_name}' with CPU inference")

    def compute_vectors(texts: List[str]):
        """Compute sparse vectors using SPLADE-style log-saturation pooling.

        Args:
            texts: List of text strings to encode.

        Returns:
            Tuple containing:
                - indices: List of lists of non-zero indices
                - values: List of lists of corresponding values
        """
        # Tokenize texts with truncation and padding
        tokens = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            tokens = tokens.to("cuda")

        # Compute sparse representations using SPLADE method
        with torch.no_grad():
            output = model(**tokens)
            logits, attention_mask = output.logits, tokens.attention_mask
            # Apply ReLU and log-saturation: log(1 + ReLU(x))
            relu_log = torch.log(1 + torch.relu(logits))
            # Mask out padding tokens
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            # Max pooling over sequence dimension
            tvecs, _ = torch.max(weighted_log, dim=1)

        # Extract non-zero indices and values for each text
        indices: List[List[int]] = []
        values: List[List[float]] = []
        for batch in tvecs:
            nz_indices = batch.nonzero(as_tuple=True)[0].tolist()
            indices.append(nz_indices)
            values.append(batch[nz_indices].tolist())

        return indices, values

    return compute_vectors


def get_sparse_encoder(
    model_name: Optional[str] = None,
    use_fastembed: bool = True,
    batch_size: int = 256,
    cache_dir: Optional[str] = None,
    threads: Optional[int] = None,
) -> Optional[Callable]:
    """Get a sparse encoder function for hybrid search.

    Factory function that creates a sparse encoder using either FastEmbed (recommended)
    or HuggingFace Transformers backend.

    Args:
        model_name: Model name or alias (e.g., "splade_pp"). If None, returns None.
        use_fastembed: Whether to use FastEmbed (ONNX-based, faster) or Transformers
            (PyTorch-based, slower) backend (default: True).
        batch_size: Number of texts to process in each batch. Only used with FastEmbed
            (default: 256).
        cache_dir: Directory for caching downloaded models. Only used with FastEmbed.
        threads: Number of threads for ONNX inference. Only used with FastEmbed.

    Returns:
        Optional[Callable]: Sparse encoder function that takes List[str] and returns
            Tuple[List[List[int]], List[List[float]]], or None if model_name is None.

    Raises:
        ImportError: If required backend (fastembed or transformers) is not installed.

    Examples:
        >>> encoder = get_sparse_encoder("splade_pp", use_fastembed=True)
        >>> indices, values = encoder(["hello world", "test document"])
    """
    if model_name is None:
        return None

    logger.debug(
        f"Initializing sparse encoder: model={model_name}, "
        f"backend={'FastEmbed' if use_fastembed else 'Transformers'}"
    )

    if use_fastembed:
        return _initialize_sparse_encoder_fastembed(
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=cache_dir,
            threads=threads,
        )
    else:
        return _initialize_sparse_encoder_transformers(model_name=model_name)