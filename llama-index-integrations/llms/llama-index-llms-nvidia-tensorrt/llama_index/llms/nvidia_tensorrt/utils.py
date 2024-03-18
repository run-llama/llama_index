import time
import uuid
from typing import Any, Dict, Optional

import numpy as np


def parse_input(
    input_text: str, tokenizer: Any, end_id: int, remove_input_padding: bool
) -> Any:
    try:
        import torch
    except ImportError:
        raise ImportError("nvidia_tensorrt requires `pip install torch`.")

    input_tokens = []

    input_tokens.append(tokenizer.encode(input_text, add_special_tokens=False))

    input_lengths = torch.tensor(
        [len(x) for x in input_tokens], dtype=torch.int32, device="cuda"
    )
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda").unsqueeze(
            0
        )
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32), end_id
        ).cuda()

    return input_ids, input_lengths


def remove_extra_eos_ids(outputs: Any) -> Any:
    outputs.reverse()
    while outputs and outputs[0] == 2:
        outputs.pop(0)
    outputs.reverse()
    outputs.append(2)
    return outputs


def get_output(
    output_ids: Any,
    input_lengths: Any,
    max_output_len: int,
    tokenizer: Any,
) -> Any:
    num_beams = output_ids.size(1)
    output_text = ""
    outputs = None
    for b in range(input_lengths.size(0)):
        for beam in range(num_beams):
            output_begin = input_lengths[b]
            output_end = input_lengths[b] + max_output_len
            outputs = output_ids[b][beam][output_begin:output_end].tolist()
            outputs = remove_extra_eos_ids(outputs)
            output_text = tokenizer.decode(outputs)

    return output_text, outputs


def generate_completion_dict(
    text_str: str, model: Any, model_path: Optional[str]
) -> Dict:
    """
    Generate a dictionary for text completion details.

    Returns:
    dict: A dictionary containing completion details.
    """
    completion_id: str = f"cmpl-{uuid.uuid4()!s}"
    created: int = int(time.time())
    model_name: str = model if model is not None else model_path
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "text": text_str,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
