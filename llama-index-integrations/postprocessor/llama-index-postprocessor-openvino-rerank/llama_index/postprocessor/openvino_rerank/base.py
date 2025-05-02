from typing import Any, List, Optional, Dict
from pathlib import Path
import numpy as np

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.utils import infer_torch_device

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForSequenceClassification

DEFAULT_COLBERT_MAX_LENGTH = 512

dispatcher = get_dispatcher(__name__)


class OpenVINORerank(BaseNodePostprocessor):
    model_id_or_path: str = Field(description="Huggingface model id or local path.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 3,
        model_id_or_path: str = "BAAI/bge-reranker-large",
        device: Optional[str] = "auto",
        model_kwargs: Dict[str, Any] = {},
        keep_retrieval_score: Optional[bool] = False,
    ):
        device = infer_torch_device() if device is None else device
        super().__init__(
            top_n=top_n,
            model_id_or_path=model_id_or_path,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
        )

        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ValueError(
                "Could not import huggingface_hub python package. "
                "Please install it with: "
                "`pip install -U huggingface_hub`."
            ) from e

        def require_model_export(
            model_id: str, revision: Any = None, subfolder: Any = None
        ) -> bool:
            model_dir = Path(model_id)
            if subfolder is not None:
                model_dir = model_dir / subfolder
            if model_dir.is_dir():
                return (
                    not (model_dir / "openvino_model.xml").exists()
                    or not (model_dir / "openvino_model.bin").exists()
                )
            hf_api = HfApi()
            try:
                model_info = hf_api.model_info(model_id, revision=revision or "main")
                normalized_subfolder = (
                    None if subfolder is None else Path(subfolder).as_posix()
                )
                model_files = [
                    file.rfilename
                    for file in model_info.siblings
                    if normalized_subfolder is None
                    or file.rfilename.startswith(normalized_subfolder)
                ]
                ov_model_path = (
                    "openvino_model.xml"
                    if subfolder is None
                    else f"{normalized_subfolder}/openvino_model.xml"
                )
                return (
                    ov_model_path not in model_files
                    or ov_model_path.replace(".xml", ".bin") not in model_files
                )
            except Exception:
                return True

        if require_model_export(model_id_or_path):
            # use remote model
            self._model = OVModelForSequenceClassification.from_pretrained(
                model_id_or_path, export=True, device=device, **model_kwargs
            )
        else:
            # use local model
            self._model = OVModelForSequenceClassification.from_pretrained(
                model_id_or_path, device=device, **model_kwargs
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINORerank"

    @staticmethod
    def create_and_save_openvino_model(
        model_name_or_path: str,
        output_path: str,
        export_kwargs: Optional[dict] = None,
    ) -> None:
        export_kwargs = export_kwargs or {}
        model = OVModelForSequenceClassification.from_pretrained(
            model_name_or_path, export=True, compile=False, **export_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(
            f"Saved OpenVINO model to {output_path}. Use it with "
            f"`embed_model = OpenVINORerank(model='{output_path}')`."
        )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model_id_or_path,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        nodes_text_list = [
            str(node.node.get_content(metadata_mode=MetadataMode.EMBED))
            for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model_id_or_path,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            query_pairs = [[query_bundle.query_str, text] for text in nodes_text_list]

            length = self._model.request.inputs[0].get_partial_shape()[1]
            if length.is_dynamic:
                input_tensors = self._tokenizer(
                    query_pairs, padding=True, truncation=True, return_tensors="pt"
                )
            else:
                input_tensors = self._tokenizer(
                    query_pairs,
                    padding="max_length",
                    max_length=length.get_length(),
                    truncation=True,
                    return_tensors="pt",
                )

            outputs = self._model(**input_tensors, return_dict=True)
            logits = outputs[0]

            if logits.shape[1] == 1:
                scores = 1 / (1 + np.exp(-logits.flatten()))
            else:
                exp_logits = np.exp(logits)
                scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes[: self.top_n]))
        return new_nodes
