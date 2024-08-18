"""LlamaPack class."""

from typing import Any, Dict, List

from llama_index.core import Settings, VectorStoreIndex, set_global_tokenizer
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document
from llama_index.llms.huggingface import HuggingFaceLLM


class ZephyrQueryEnginePack(BaseLlamaPack):
    def __init__(self, documents: List[Document]) -> None:
        """Init params."""
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Dependencies missing, run "
                "`pip install torch transformers accelerate bitsandbytes`"
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        try:
            llm = HuggingFaceLLM(
                model_name="HuggingFaceH4/zephyr-7b-beta",
                tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
                query_wrapper_prompt=PromptTemplate(
                    "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
                ),
                context_window=3900,
                max_new_tokens=256,
                model_kwargs={"quantization_config": quantization_config},
                generate_kwargs={
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                },
                device_map="auto",
            )
        except Exception:
            print(
                "Failed to load and quantize model, likely due to CUDA being missing. "
                "Loading full precision model instead."
            )
            llm = HuggingFaceLLM(
                model_name="HuggingFaceH4/zephyr-7b-beta",
                tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
                query_wrapper_prompt=PromptTemplate(
                    "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
                ),
                context_window=3900,
                max_new_tokens=256,
                generate_kwargs={
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                },
                device_map="auto",
            )

        # set tokenizer for proper token counting
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        set_global_tokenizer(tokenizer.encode)

        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-base-en-v1.5"

        self.llm = llm
        self.index = VectorStoreIndex.from_documents(documents)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        query_engine = self.index.as_query_engine(**kwargs)
        return query_engine.query(query_str)
