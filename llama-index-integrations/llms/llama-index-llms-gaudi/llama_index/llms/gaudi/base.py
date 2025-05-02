import logging
from typing import Any, Callable, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field
from llama_index.llms.huggingface.base import HuggingFaceLLM
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.prompts.base import PromptTemplate

from llama_index.llms.gaudi.utils import initialize_model

DEFAULT_HUGGINGFACE_MODEL = "Intel/neural-chat-7b-v3-1"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GaudiLLM(HuggingFaceLLM):
    r"""GaudiLLM LLM.

    Examples:
        `pip install llama-index-llms-gaudi`

        ```python
        from llama_index.llms.gaudi import GaudiLLM
        import argparse
        import os, logging

        def setup_parser(parser):
            # Arguments management
            parser.add_argument(
                "--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu"
            )
            parser.add_argument(
                "--model_name_or_path",
                default=None,
                type=str,
                # required=True,
                help="Path to pre-trained model (on the HF Hub or locally).",
            )
            parser.add_argument(
                "--bf16",
                default=True,
                action="store_true",
                help="Whether to perform generation in bf16 precision.",
            )
            parser.add_argument(
                "--max_new_tokens", type=int, default=100, help="Number of tokens to generate."
            )
            parser.add_argument(
                "--max_input_tokens",
                type=int,
                default=0,
                help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
                    if == 0, then truncate to 16 (original default) \
                    if < 0, then do not truncate, use full input prompt",
            )
            parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
            parser.add_argument(
                "--warmup",
                type=int,
                default=3,
                help="Number of warmup iterations for benchmarking.",
            )
            parser.add_argument(
                "--n_iterations",
                type=int,
                default=5,
                help="Number of inference iterations for benchmarking.",
            )
            parser.add_argument(
                "--local_rank", type=int, default=0, metavar="N", help="Local process rank."
            )
            parser.add_argument(
                "--use_kv_cache",
                default=True,
                action="store_true",
                help="Whether to use the key/value cache for decoding. It should speed up generation.",
            )
            parser.add_argument(
                "--use_hpu_graphs",
                default=True,
                action="store_true",
                help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
            )
            parser.add_argument(
                "--dataset_name",
                default=None,
                type=str,
                help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
            )
            parser.add_argument(
                "--column_name",
                default=None,
                type=str,
                help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
            )
            parser.add_argument(
                "--do_sample",
                action="store_true",
                help="Whether to use sampling for generation.",
            )
            parser.add_argument(
                "--num_beams",
                default=1,
                type=int,
                help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
            )
            parser.add_argument(
                "--trim_logits",
                action="store_true",
                help="Calculate logits only for the last token to save memory in the first step.",
            )
            parser.add_argument(
                "--seed",
                default=27,
                type=int,
                help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
            )
            parser.add_argument(
                "--profiling_warmup_steps",
                default=0,
                type=int,
                help="Number of steps to ignore for profiling.",
            )
            parser.add_argument(
                "--profiling_steps",
                default=0,
                type=int,
                help="Number of steps to capture for profiling.",
            )
            parser.add_argument(
                "--profiling_record_shapes",
                default=False,
                type=bool,
                help="Record shapes when enabling profiling.",
            )
            parser.add_argument(
                "--prompt",
                default=None,
                type=str,
                nargs="*",
                help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
            )
            parser.add_argument(
                "--bad_words",
                default=None,
                type=str,
                nargs="+",
                help="Optional argument list of words that are not allowed to be generated.",
            )
            parser.add_argument(
                "--force_words",
                default=None,
                type=str,
                nargs="+",
                help="Optional argument list of words that must be generated.",
            )
            parser.add_argument(
                "--assistant_model",
                default=None,
                type=str,
                help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
            )
            parser.add_argument(
                 "--peft_model",
                default=None,
                type=str,
                help="Optional argument to give a path to a PEFT model.",
            )
            parser.add_argument("--num_return_sequences", type=int, default=1)
            parser.add_argument(
                "--token",
                default=None,
                type=str,
                help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
            )
            parser.add_argument(
                "--model_revision",
                default="main",
                type=str,
                help="The specific model version to use (can be a branch name, tag name or commit id).",
            )
            parser.add_argument(
                "--attn_softmax_bf16",
                action="store_true",
                help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
                "is also running in lower precision.",
            )
            parser.add_argument(
                "--output_dir",
                default=None,
                type=str,
                help="Output directory to store results in.",
            )
            parser.add_argument(
                "--bucket_size",
                default=-1,
                type=int,
                help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
                    then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
                    we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
            )
            parser.add_argument(
                "--bucket_internal",
                action="store_true",
                help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
            )
            parser.add_argument(
                "--dataset_max_samples",
                default=-1,
                type=int,
                help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
            )
            parser.add_argument(
                "--limit_hpu_graphs",
                action="store_true",
                help="Skip HPU Graph usage for first token to save memory",
            )
            parser.add_argument(
                "--reuse_cache",
                action="store_true",
                help="Whether to reuse key/value cache for decoding. It should save memory.",
            )
            parser.add_argument(
                "--verbose_workers",
                action="store_true",
                help="Enable output from non-master workers",
            )
            parser.add_argument(
                "--simulate_dyn_prompt",
                default=None,
                type=int,
                nargs="*",
                help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
            )
            parser.add_argument(
                "--reduce_recompile",
                action="store_true",
                help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
            )
            parser.add_argument(
                "--use_flash_attention",
                action="store_true",
                help="Whether to enable Habana Flash Attention, provided that the model supports it.",
            )
            parser.add_argument(
                "--flash_attention_recompute",
                action="store_true",
                help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
            )
            parser.add_argument(
                "--flash_attention_causal_mask",
                action="store_true",
                help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
            )
            parser.add_argument(
                "--flash_attention_fast_softmax",
                action="store_true",
                help="Whether to enable Habana Flash Attention in fast softmax mode.",
            )
            parser.add_argument(
                "--book_source",
                action="store_true",
                help="Whether to use project Guttenberg books data as input. Useful for testing large sequence lengths.",
            )
            parser.add_argument(
                "--torch_compile",
                action="store_true",
                help="Whether to use torch compiled model or not.",
            )
            parser.add_argument(
                "--ignore_eos",
                default=True,
                action=argparse.BooleanOptionalAction,
                help="Whether to ignore eos, set False to disable it",
            )
            parser.add_argument(
                "--temperature",
                default=1.0,
                type=float,
                help="Temperature value for text generation",
            )
            parser.add_argument(
                "--top_p",
                default=1.0,
                type=float,
                help="Top_p value for generating text via sampling",
            )
            parser.add_argument(
                "--const_serialization_path",
                "--csp",
                type=str,
                help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
            )
            parser.add_argument(
                "--disk_offload",
                action="store_true",
                help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
            )
            parser.add_argument(
                "--trust_remote_code",
                action="store_true",
                help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
            )
            args = parser.parse_args()

            if args.torch_compile:
                args.use_hpu_graphs = False

            if not args.use_hpu_graphs:
                args.limit_hpu_graphs = False

            args.quant_config = os.getenv("QUANT_CONFIG", "")
            if args.quant_config == "" and args.disk_offload:
                logger.warning(
                    "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
                )
            return args

        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

        import torch
        from llama_index.core.prompts import PromptTemplate
        from llama_index.llms.optimum-intel import GaudiLLM

        parser = argparse.ArgumentParser(description="GaudiLLM Basic Usage Example")
        args = setup_parser(parser)
        args.model_name_or_path = "HuggingFaceH4/zephyr-7b-alpha"

        llm = GaudiLLM(
            args=args,
            logger=logger,
            model_name="HuggingFaceH4/zephyr-7b-alpha",
            tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
            query_wrapper_prompt=PromptTemplate(
                "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
            ),
            context_window=3900,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            messages_to_prompt=messages_to_prompt,
            device_map="auto",
        )

        response = llm.complete("What is the meaning of life?")
        print(str(response))
        ```
    """

    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )

    def __init__(
        self,
        args,
        logger,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: Optional[str] = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""
        model_kwargs = model_kwargs or {}

        model, _, tokenizer, _ = initialize_model(args, logger)

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=tokenizer_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GaudiLLM"
