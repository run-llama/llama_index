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

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
)
from llama_index.llms.nvidia_tensorrt.utils import (
    generate_completion_dict,
    get_output,
    parse_input,
)
from transformers import AutoTokenizer

EOS_TOKEN = 2
PAD_TOKEN = 2


class LocalTensorRTLLM(CustomLLM):
    r"""
    Local TensorRT LLM.

    [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference
    efficiently on NVIDIA GPUs.

    Since TensorRT-LLM is a SDK for interacting with local models in process there are a few environment steps that must be followed to ensure that the TensorRT-LLM setup can be used.

    1. Nvidia Cuda 12.2 or higher is currently required to run TensorRT-LLM
    2. Install `tensorrt_llm` via pip with `pip3 install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com`
    3. For this example we will use Llama2. The Llama2 model files need to be created via scripts following the instructions
    (https://github.com/NVIDIA/trt-llm-rag-windows/blob/release/1.0/README.md#building-trt-engine)
        * The following files will be created from following the stop above
        * `Llama_float16_tp1_rank0.engine`: The main output of the build script, containing the executable graph of operations with the model weights embedded.
        * `config.json`: Includes detailed information about the model, like its general structure and precision, as well as information about which plug-ins were incorporated into the engine.
        * `model.cache`: Caches some of the timing and optimization information from model compilation, making successive builds quicker.
    4. `mkdir model`
    5. Move all of the files mentioned above to the model directory.

    Examples:
        `pip install llama-index-llms-nvidia-tensorrt`

        ```python
        from llama_index.llms.nvidia_tensorrt import LocalTensorRTLLM


        def completion_to_prompt(completion):
            return f"<s> [INST] {completion} [/INST] "

        def messages_to_prompt(messages):
            content = ""
            for message in messages:
                content += str(message) + "\n"
            return f"<s> [INST] {content} [/INST] "

        llm = LocalTensorRTLLM(
            model_path="./model",
            engine_name="llama_float16_tp1_rank0.engine",
            tokenizer_dir="meta-llama/Llama-2-13b-chat",
            completion_to_prompt=completion_to_prompt,
            messages_to_prompt=messages_to_prompt,
        )

        resp = llm.complete("Who is Paul Graham?")
        print(str(resp))
        ```

    """

    model_path: Optional[str] = Field(description="The path to the trt engine.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    messages_to_prompt: Callable = Field(
        description="The function to convert messages to a prompt.", exclude=True
    )
    completion_to_prompt: Callable = Field(
        description="The function to convert a completion to a prompt.", exclude=True
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(description="Whether to print verbose output.")

    _model: Any = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _max_new_tokens = PrivateAttr()
    _sampling_config = PrivateAttr()
    _verbose = PrivateAttr()

    def __init__(
        self,
        model_path: Optional[str] = None,
        engine_name: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelConfig, SamplingConfig
        except ImportError:
            print(
                "Unable to import `tensorrt_llm` module. Please ensure you have\
                  `tensorrt_llm` installed in your environment. You can run\
                  `pip3 install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com` to install."
            )

        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})
        max_new_tokens = max_new_tokens
        verbose = verbose
        # check if model is cached
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                engine_dir = model_path
                engine_dir_path = Path(engine_dir)
                config_path = engine_dir_path / "config.json"

                # config function
                with open(config_path) as f:
                    config = json.load(f)
                use_gpt_attention_plugin = config["plugin_config"][
                    "gpt_attention_plugin"
                ]
                remove_input_padding = config["plugin_config"]["remove_input_padding"]
                tp_size = config["builder_config"]["tensor_parallel"]
                pp_size = 1
                if "pipeline_parallel" in config["builder_config"]:
                    pp_size = config["builder_config"]["pipeline_parallel"]
                world_size = tp_size * pp_size
                assert world_size == tensorrt_llm.mpi_world_size(), (
                    f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
                )
                num_heads = config["builder_config"]["num_heads"] // tp_size
                hidden_size = config["builder_config"]["hidden_size"] // tp_size
                vocab_size = config["builder_config"]["vocab_size"]
                num_layers = config["builder_config"]["num_layers"]
                num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)
                paged_kv_cache = config["plugin_config"]["paged_kv_cache"]
                if config["builder_config"].get("multi_query_mode", False):
                    tensorrt_llm.logger.warning(
                        "`multi_query_mode` config is deprecated. Please rebuild the engine."
                    )
                    num_kv_heads = 1
                num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

                model_config = ModelConfig(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    gpt_attention_plugin=use_gpt_attention_plugin,
                    paged_kv_cache=paged_kv_cache,
                    remove_input_padding=remove_input_padding,
                    max_batch_size=config["builder_config"]["max_batch_size"],
                )

                assert pp_size == 1, (
                    "Python runtime does not support pipeline parallelism"
                )
                world_size = tp_size * pp_size

                runtime_rank = tensorrt_llm.mpi_rank()
                runtime_mapping = tensorrt_llm.Mapping(
                    world_size, runtime_rank, tp_size=tp_size, pp_size=pp_size
                )

                # TensorRT-LLM must run on a GPU.
                assert torch.cuda.is_available(), (
                    "LocalTensorRTLLM requires a Nvidia CUDA enabled GPU to operate"
                )
                torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False)
                sampling_config = SamplingConfig(
                    end_id=EOS_TOKEN,
                    pad_id=PAD_TOKEN,
                    num_beams=1,
                    temperature=temperature,
                )

                serialize_path = engine_dir_path / (engine_name if engine_name else "")
                with open(serialize_path, "rb") as f:
                    engine_buffer = f.read()
                decoder = tensorrt_llm.runtime.GenerationSession(
                    model_config, engine_buffer, runtime_mapping, debug_mode=False
                )
                model = decoder

        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )
        self._model = model
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._sampling_config = sampling_config
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LocalTensorRTLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        try:
            import torch
        except ImportError:
            raise ImportError("nvidia_tensorrt requires `pip install torch`.")

        self.generate_kwargs.update({"stream": False})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        input_text = prompt
        input_ids, input_lengths = parse_input(
            input_text, self._tokenizer, EOS_TOKEN, self._model_config
        )

        max_input_length = torch.max(input_lengths).item()
        self._model.setup(
            input_lengths.size(0), max_input_length, self._max_new_tokens, 1
        )  # beam size is set to 1
        if self._verbose:
            start_time = time.time()

        output_ids = self._model.decode(input_ids, input_lengths, self._sampling_config)
        torch.cuda.synchronize()

        elapsed_time = -1.0
        if self._verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time

        output_txt, output_token_ids = get_output(
            output_ids, input_lengths, self._max_new_tokens, self._tokenizer
        )

        if self._verbose:
            print(f"Input context length  : {input_ids.shape[1]}")
            print(f"Inference time        : {elapsed_time:.2f} seconds")
            print(f"Output context length : {len(output_token_ids)} ")
            print(
                f"Inference token/sec   : {(len(output_token_ids) / elapsed_time):2f}"
            )

        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()

        return CompletionResponse(
            text=output_txt,
            raw=generate_completion_dict(output_txt, self._model, self.model_path),
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError(
            "Nvidia TensorRT-LLM does not currently support streaming completion."
        )
