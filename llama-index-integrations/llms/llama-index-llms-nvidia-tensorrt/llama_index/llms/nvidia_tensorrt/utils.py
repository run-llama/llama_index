# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
