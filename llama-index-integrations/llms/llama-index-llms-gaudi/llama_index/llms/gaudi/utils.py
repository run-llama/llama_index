# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import copy
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import check_min_version

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from optimum.habana.utils import (
    check_habana_frameworks_version,
    check_optimum_habana_min_version,
    get_habana_frameworks_version,
    set_seed,
)


def adjust_batch(batch, size):
    curr_size = batch["input_ids"].shape[1]
    if curr_size >= size:
        adjusted_batch = {
            "input_ids": batch["input_ids"][:, :size],
            "attention_mask": batch["attention_mask"][:, :size],
        }
    else:
        adjusted_batch = {}
        for k in batch:
            last_colm = batch[k][:, -1]
            expanded = last_colm.tile((size - curr_size, 1)).T
            adjusted_batch[k] = torch.concat([batch[k], expanded], 1)
    assert adjusted_batch["input_ids"].shape[1] == size
    assert adjusted_batch["attention_mask"].shape[1] == size
    return adjusted_batch


def override_print(enable):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def override_logger(logger, enable):
    logger_info = logger.info

    def info(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            logger_info(*args, **kwargs)

    logger.info = info


def count_hpu_graphs():
    return len(glob.glob(".graph_dumps/*PreGraph*"))


def override_prints(enable, logger):
    override_print(enable)
    override_logger(logger, enable)


def setup_distributed(args):
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "0"))
    args.global_rank = int(os.getenv("RANK", "0"))


def setup_inference(args, model):
    import habana_frameworks.torch.core as htcore

    habana_version = get_habana_frameworks_version()

    print("Initializing inference mode")
    # Keeping the if-else here for back compat. TODO remove later
    if habana_version.major >= 1 and habana_version.minor >= 16:
        htcore.hpu_initialize(model, mark_only_scales_as_const=True)
    else:
        const_marking = os.getenv("ENABLE_CONST_MARKING", "True")
        if const_marking == "True":
            htcore.hpu_initialize(model)
    return model


def setup_const_serialization(const_serialization_path):
    import uuid

    const_serialization_path = os.path.join(const_serialization_path + uuid.uuid4().hex)
    os.makedirs(const_serialization_path)
    from habana_frameworks.torch.hpu import enable_const_section_serialization

    print(f"Serializing const params to {const_serialization_path}")
    enable_const_section_serialization(const_serialization_path, True)


def setup_env(args):
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.34.0")
    check_optimum_habana_min_version("1.9.0.dev0")
    # TODO: SW-167588 - WA for memory issue in hqt prep_model
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")

    if args.global_rank == 0 and not args.torch_compile:
        os.environ.setdefault("GRAPH_VISUALIZATION", "true")
        shutil.rmtree(".graph_dumps", ignore_errors=True)

    if args.world_size > 0:
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    if (
        args.use_hpu_graphs
        and args.limit_hpu_graphs
        and not args.reuse_cache
        and args.bucket_internal
    ):
        # Based upon above conditions and below env variable,
        # we can call HPU graphs clear_inputs().
        os.environ.setdefault("PT_HPUGRAPH_DISABLE_TENSOR_CACHE", "1")

    # Tweak generation so that it runs faster on Gaudi
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()


def setup_device(args):
    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore

        if args.quant_config:
            htcore.hpu_set_env()
    return torch.device(args.device)


# patching LinearAllreduce to use ScopedLinearAllReduce
def patch_scoped_linear_all_reduce(model):
    from deepspeed.module_inject.layers import LinearAllreduce

    from optimum.habana.transformers.models.modeling_all_models import (
        ScopedLinearAllReduce,
    )

    for name, module in model.named_children():
        if type(module) is LinearAllreduce:
            SL = ScopedLinearAllReduce(mod=module)
            setattr(model, name, SL)
        patch_scoped_linear_all_reduce(module)


def get_torch_compiled_model(model):
    model.model = torch.compile(
        model.model, backend="hpu_backend", options={"keep_input_mutations": True}
    )
    return model


def setup_model(args, model_dtype, model_kwargs, logger):
    logger.info("Single-device run.")
    if args.assistant_model is None:
        assistant_model = None
    else:
        logger.info(f"Using asssitant model {args.assistant_model}.")
    if args.disk_offload:
        from accelerate import infer_auto_device_map, init_empty_weights

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        max_memory = {"cpu": "10GiB"}
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, dtype=model_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            offload_folder="/tmp/offload_folder/",
            offload_state_dict=True,
            torch_dtype=model_dtype,
            **model_kwargs,
        )
    else:
        if args.assistant_model is not None:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                args.assistant_model, torch_dtype=model_dtype, **model_kwargs
            )
        if args.peft_model is not None:
            model = peft_model(args, model_dtype, logger, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
            )
    if args.quant_config:
        import habana_quantization_toolkit

        habana_quantization_toolkit.prep_model(model)
        if args.assistant_model is not None:
            habana_quantization_toolkit.quantize_model(assistant_model)

    model = model.eval().to(args.device)
    if args.assistant_model is not None:
        assistant_model = assistant_model.eval().to(args.device)

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        from optimum.habana.transformers.trainer import _is_peft_model

        if (
            check_habana_frameworks_version("1.13.0")
            and model.config.model_type == "falcon"
        ):
            model = wrap_in_hpu_graph(model, hash_with_views=False)
        else:
            model = wrap_in_hpu_graph(model)
        if args.assistant_model is not None:
            assistant_model = wrap_in_hpu_graph(assistant_model)
        if _is_peft_model(model):
            model.base_model = wrap_in_hpu_graph(model.base_model)

    if args.torch_compile and model.config.model_type == "llama":
        model = get_torch_compiled_model(model)
        # if args.assistant_model is not None:
        #     assistant_model = get_torch_compiled_model(assistant_model)
    return model, assistant_model


def setup_distributed_model(args, model_dtype, model_kwargs, logger):
    import deepspeed

    logger.info("DeepSpeed is enabled.")
    deepspeed.init_distributed(dist_backend="hccl")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
    )
    load_to_meta = model_on_meta(config)

    if args.assistant_model is None:
        assistant_model = None
    else:
        logger.info(f"Using asssitant model {args.assistant_model}.")

    if load_to_meta:
        # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
        with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)

        # Model loaded to meta is managed differently
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")

        # For PEFT models, write the merged model on disk to be able to load it on the meta device
        if args.peft_model is not None:
            merged_model_dir = "/tmp/text_generation_merged_peft_model"
            if args.local_rank == 0:
                if Path(merged_model_dir).is_dir():
                    shutil.rmtree(merged_model_dir)
                peft_model(args, model_dtype, logger, **model_kwargs).save_pretrained(
                    merged_model_dir
                )
            torch.distributed.barrier()

        write_checkpoints_json(
            merged_model_dir
            if args.peft_model is not None
            else args.model_name_or_path,
            args.local_rank,
            checkpoints_json,
            token=args.token,
        )
    else:
        # TODO: revisit placement on CPU when auto-injection is possible
        with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
            if args.peft_model is not None:
                model = peft_model(args, model_dtype, logger, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
                )
    model.eval()

    if args.assistant_model is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            args.assistant_model, torch_dtype=model_dtype, **model_kwargs
        ).eval()

    # Initialize the model
    ds_inference_kwargs = {"dtype": model_dtype}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": args.world_size}
    ds_inference_kwargs["enable_cuda_graph"] = args.use_hpu_graphs
    ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
    if load_to_meta:
        ds_inference_kwargs["checkpoint"] = checkpoints_json.name

    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    model = model.module
    if model.config.model_type in ["llama", "falcon", "qwen2"]:
        patch_scoped_linear_all_reduce(model)

    if args.quant_config:
        import habana_quantization_toolkit

        habana_quantization_toolkit.prep_model(model)
        if args.assistant_model is not None:
            habana_quantization_toolkit.prep_model(assistant_model)

    if args.torch_compile and model.config.model_type == "llama":
        model = get_torch_compiled_model(model)
        # if args.assistant_model is not None:
        #     assistant_model = get_torch_compiled_model(assistant_model)
    return model, assistant_model


def peft_model(args, model_dtype, logger, **model_kwargs):
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        raise ImportError(
            "The `peft` package is not installed, please run: `pip install peft`."
        )
    from peft import AutoPeftModelForCausalLM
    from peft.config import PeftConfigMixin

    base_model_name = PeftConfigMixin.from_pretrained(
        args.peft_model,
        token=model_kwargs["token"] if "token" in model_kwargs else None,
    ).base_model_name_or_path

    base_model_is_local = Path(base_model_name).is_dir()
    if not base_model_is_local:
        # Check if the base model path to a remote repository on the HF Hub exists
        from huggingface_hub import list_repo_files

        try:
            list_repo_files(base_model_name)
            base_model_is_remote = True
        except Exception:
            base_model_is_remote = False

    if base_model_is_local or base_model_is_remote:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.peft_model, torch_dtype=model_dtype, **model_kwargs
        )
    else:
        # Since the base model doesn't exist locally nor remotely, use `args.model_name_or_path` as the base model
        logger.warning(
            f"The base model `{base_model_name}` of the LoRA configuration associated"
            f" to `{args.peft_model}` does not exist locally or remotely. Using "
            f"`--model_name_or_path {args.model_name_or_path}` as a fall back for the base model."
        )
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
        )
        model = PeftModel.from_pretrained(
            model, args.peft_model, torch_dtype=model_dtype, **model_kwargs
        )
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        if model_dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)
        return model
    else:
        from optimum.habana.peft.peft_model import (
            gaudi_generate,
            gaudi_prepare_inputs_for_generation,
        )

        model.__class__.generate = gaudi_generate
        model.__class__.prepare_inputs_for_generation = (
            gaudi_prepare_inputs_for_generation
        )
        return model


def setup_tokenizer(args, model, assistant_model):
    tokenizer_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.bad_words is not None or args.force_words is not None:
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, **tokenizer_kwargs
    )
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if model.config.model_type == "llama":
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        if assistant_model is not None:
            assistant_model.generation_config.pad_token_id = 0
            assistant_model.generation_config.bos_token_id = 1
            assistant_model.generation_config.eos_token_id = 2
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
    if model.config.model_type == "persimmon":
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if assistant_model is not None:
            assistant_model.generation_config.pad_token_id = (
                assistant_model.generation_config.eos_token_id
            )
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if assistant_model is not None:
            assistant_model.generation_config.pad_token_id = (
                assistant_model.generation_config.eos_token_id
            )

    return tokenizer, model, assistant_model


def setup_generation_config(args, model, assistant_model, tokenizer):
    bad_words_ids = None
    force_words_ids = None
    if args.bad_words is not None:
        bad_words_ids = [
            tokenizer.encode(bad_word, add_special_tokens=False)
            for bad_word in args.bad_words
        ]
    if args.force_words is not None:
        force_words_ids = [
            tokenizer.encode(force_word, add_special_tokens=False)
            for force_word in args.force_words
        ]

    is_optimized = model_is_optimized(model.config)

    # Generation configuration
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = args.use_kv_cache
    generation_config.static_shapes = is_optimized and assistant_model is None
    generation_config.bucket_size = args.bucket_size if is_optimized else -1
    generation_config.bucket_internal = args.bucket_internal
    generation_config.do_sample = args.do_sample
    generation_config.num_beams = args.num_beams
    generation_config.bad_words_ids = bad_words_ids
    generation_config.force_words_ids = force_words_ids
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.trim_logits = args.trim_logits
    generation_config.attn_softmax_bf16 = args.attn_softmax_bf16
    generation_config.limit_hpu_graphs = args.limit_hpu_graphs
    generation_config.reuse_cache = args.reuse_cache
    generation_config.reduce_recompile = args.reduce_recompile
    if generation_config.reduce_recompile:
        assert generation_config.bucket_size > 0
    generation_config.use_flash_attention = args.use_flash_attention
    generation_config.flash_attention_recompute = args.flash_attention_recompute
    generation_config.flash_attention_causal_mask = args.flash_attention_causal_mask
    generation_config.flash_attention_fast_softmax = args.flash_attention_fast_softmax
    generation_config.trust_remote_code = args.trust_remote_code

    return generation_config


def exclude_hpu_graph_configs(args):
    # Excluded configs for batch size 1 for hpu graph
    if args.batch_size == 1 and args.limit_hpu_graphs:
        if (
            "falcon-180B" in args.model_name_or_path
            or "falcon-180b" in args.model_name_or_path
        ):
            return False
        if args.world_size == 2 or args.world_size == 4 or args.world_size == 8:
            if args.quant_config:
                if args.max_input_tokens >= 8192 and args.max_new_tokens >= 128:
                    return False
            else:
                if args.max_input_tokens >= 4096 and args.max_new_tokens >= 128:
                    return False
        return True
    else:
        return False


def initialize_model(args, logger):
    init_start = time.perf_counter()
    setup_distributed(args)
    if exclude_hpu_graph_configs(args):
        args.limit_hpu_graphs = False
    override_prints(args.global_rank == 0 or args.verbose_workers, logger)
    setup_env(args)
    setup_device(args)
    set_seed(args.seed)
    get_repo_root(args.model_name_or_path, local_rank=args.local_rank, token=args.token)
    if args.assistant_model is not None:
        get_repo_root(
            args.assistant_model, local_rank=args.local_rank, token=args.token
        )
    use_deepspeed = False
    if use_deepspeed or args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float
        args.attn_softmax_bf16 = False

    model_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.trust_remote_code:
        logger.warning(
            "`trust_remote_code` is set, there is no guarantee this model works properly and it may fail"
        )

    model, assistant_model = (
        setup_model(args, model_dtype, model_kwargs, logger)
        if not use_deepspeed
        else setup_distributed_model(args, model_dtype, model_kwargs, logger)
    )
    tokenizer, model, assistant_model = setup_tokenizer(args, model, assistant_model)
    generation_config = setup_generation_config(args, model, assistant_model, tokenizer)

    if args.const_serialization_path:
        setup_const_serialization(args.const_serialization_path)
    if args.quant_config:
        model = setup_inference(args, model)
    init_end = time.perf_counter()
    logger.info(f"Args: {args}")
    logger.info(
        f"device: {args.device}, n_hpu: {args.world_size}, bf16: {model_dtype == torch.bfloat16}"
    )
    logger.info(f"Model initialization took {(init_end - init_start):.3f}s")
    return model, assistant_model, tokenizer, generation_config
