from datetime import time
from typing import Dict, Optional, Tuple, Generator, Union, Callable, Any

import mlx.core as mx
import mlx.nn as nn
from .tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.

    """
    if len(generated_tokens) > 0:
        indices = mx.array([list(generated_tokens)])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.

    Returns:
        token selected based on the top-p criterion.

    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))

    return sorted_indices.squeeze(0)[sorted_token]


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty (default 20).
        top_p (float, optional): Nulceus sampling, higher means model considers more less likely words

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.

    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        logits = logits + logit_bias if logit_bias else logits
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            else:
                token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = prompt
    cache = None

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    def _step(y):
        nonlocal cache, repetition_context
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, prob = sample(logits)
            repetition_context.append(y.item())
        else:
            y, prob = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        return y, prob

    y, p = _step(y)

    mx.async_eval(y)
    while True:
        next_y, next_p = _step(y)
        mx.async_eval(next_y)
        yield y.item(), p
        y, p = next_y, next_p


def gen_full(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    formatter: Optional[Callable] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> str:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.

    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()

    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
            logit_bias,
        ),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

    detokenizer.finalize()

    return detokenizer.text


def gen_stream(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    formatter: Optional[Callable] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> str:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.

    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()

    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
            logit_bias,
        ),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)
        if formatter:
            # We have to finalize so that the prob corresponds to the last segment
            detokenizer.finalize()
            formatter(detokenizer.last_segment, prob.item())
        else:
            yield detokenizer.last_segment
