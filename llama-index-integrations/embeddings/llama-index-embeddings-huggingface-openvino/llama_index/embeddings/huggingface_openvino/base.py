from typing import Any, List, Optional, Dict
from pathlib import Path

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.core.schema import ImageType
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface.utils import format_query, format_text
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from optimum.intel.openvino import (
    OVModelForFeatureExtraction,
    OVModelOpenCLIPVisual,
    OVModelOpenCLIPText,
)
from transformers import AutoTokenizer
from PIL import Image


class OpenVINOEmbedding(BaseEmbedding):
    model_id_or_path: str = Field(description="Huggingface model id or local path.")
    max_length: int = Field(description="Maximum length of input.")
    pooling: str = Field(description="Pooling strategy. One of ['cls', 'mean'].")
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for huggingface files.", default=None
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    def __init__(
        self,
        model_id_or_path: str = "BAAI/bge-m3",
        pooling: str = "cls",
        max_length: Optional[int] = None,
        normalize: bool = True,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        model_kwargs: Dict[str, Any] = {},
        device: Optional[str] = "auto",
    ):
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
            model = model or OVModelForFeatureExtraction.from_pretrained(
                model_id_or_path, export=True, device=device, **model_kwargs
            )
        else:
            # use local model
            model = model or OVModelForFeatureExtraction.from_pretrained(
                model_id_or_path, device=device, **model_kwargs
            )
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id_or_path)

        if max_length is None:
            try:
                max_length = int(model.config.max_position_embeddings)
            except Exception:
                raise ValueError(
                    "Unable to find max_length from model config. "
                    "Please provide max_length."
                )
            try:
                max_length = min(max_length, int(tokenizer.model_max_length))
            except Exception as exc:
                print(f"An error occurred while retrieving tokenizer max length: {exc}")

        if pooling not in ["cls", "mean"]:
            raise ValueError(f"Pooling {pooling} not supported.")

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager or CallbackManager([]),
            model_id_or_path=model_id_or_path,
            max_length=max_length,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )
        self._device = device
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINOEmbedding"

    @staticmethod
    def create_and_save_openvino_model(
        model_name_or_path: str,
        output_path: str,
        export_kwargs: Optional[dict] = None,
    ) -> None:
        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction
            from transformers import AutoTokenizer
            from optimum.exporters.openvino.convert import export_tokenizer

        except ImportError:
            raise ImportError(
                "OpenVINO Embedding requires transformers and optimum to be installed.\n"
                "Please install transformers with "
                "`pip install transformers optimum[openvino]`."
            )

        export_kwargs = export_kwargs or {}
        model = OVModelForFeatureExtraction.from_pretrained(
            model_name_or_path, export=True, compile=False, **export_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        export_tokenizer(tokenizer, output_path)
        print(
            f"Saved OpenVINO model to {output_path}. Use it with "
            f"`embed_model = OpenVINOEmbedding(model_id_or_path='{output_path}')`."
        )

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        import torch

        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _cls_pooling(self, model_output: list) -> Any:
        """Use the CLS token as the pooling token."""
        return model_output[0][:, 0]

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        length = self._model.request.inputs[0].get_partial_shape()[1]
        if length.is_dynamic:
            encoded_input = self._tokenizer(
                sentences,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoded_input = self._tokenizer(
                sentences,
                padding="max_length",
                max_length=length.get_length(),
                truncation=True,
                return_tensors="pt",
            )

        model_output = self._model(**encoded_input)

        if self.pooling == "cls":
            embeddings = self._cls_pooling(model_output)
        else:
            embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

        if self.normalize:
            import torch

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query = format_query(query, self.model_name, self.query_instruction)
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text = format_text(text, self.model_name, self.text_instruction)
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        texts = [
            format_text(text, self.model_name, self.text_instruction) for text in texts
        ]
        return self._embed(texts)


class OpenVINOClipEmbedding(MultiModalEmbedding):
    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)

    _visual_model: Any = PrivateAttr()
    _text_model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINOClipEmbedding"

    @staticmethod
    def create_and_save_openvino_model(
        model_name_or_path: str,
        output_path: str,
        export_kwargs: Optional[dict] = None,
    ) -> None:
        try:
            from optimum.intel.openvino import (
                OVModelOpenCLIPForZeroShotImageClassification,
            )
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "OpenVINO Clip requires transformers and optimum to be installed.\n"
                "Please install transformers with "
                "`pip install transformers optimum-intel[openvino]`."
            )

        export_kwargs = export_kwargs or {}
        model = OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
            model_name_or_path, export=True, compile=False, **export_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(
            f"Saved OpenVINO model to {output_path}. Use it with "
            f"`embed_model = OpenVINOClipEmbedding(model_id_or_path='{output_path}')`."
        )

    def __init__(
        self,
        model_id_or_path: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        visual_model: Optional[Any] = None,
        text_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_kwargs: Dict[str, Any] = {},
        device: Optional[str] = "auto",
        **kwargs: Any,
    ):
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
                    not (model_dir / "openvino_model_vision.xml").exists()
                    or not (model_dir / "openvino_model_vision.bin").exists()
                    or not (model_dir / "openvino_model_text.xml").exists()
                    or not (model_dir / "openvino_model_text.bin").exists()
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
                visual_ov_model_path = (
                    "openvino_model_vision.xml"
                    if subfolder is None
                    else f"{normalized_subfolder}/openvino_model_vision.xml"
                )
                text_ov_model_path = (
                    "openvino_model_text.xml"
                    if subfolder is None
                    else f"{normalized_subfolder}/openvino_model_text.xml"
                )
                return (
                    text_ov_model_path not in model_files
                    or text_ov_model_path.replace(".xml", ".bin") not in model_files
                    or visual_ov_model_path not in model_files
                    or visual_ov_model_path.replace(".xml", ".bin") not in model_files
                )
            except Exception:
                return True

        if require_model_export(model_id_or_path):
            # use remote model
            visual_model = visual_model or OVModelOpenCLIPVisual.from_pretrained(
                model_id_or_path, export=True, device=device, **model_kwargs
            )
            text_model = text_model or OVModelOpenCLIPText.from_pretrained(
                model_id_or_path, export=True, device=device, **model_kwargs
            )
        else:
            # use local model
            visual_model = visual_model or OVModelOpenCLIPVisual.from_pretrained(
                model_id_or_path, device=device, **model_kwargs
            )
            text_model = text_model or OVModelOpenCLIPText.from_pretrained(
                model_id_or_path, device=device, **model_kwargs
            )
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size}  must be > 0.")

        processor_inputs = {
            "is_train": False,
            "image_size": (
                visual_model.config.vision_config.image_size,
                visual_model.config.vision_config.image_size,
            ),
        }
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id_or_path)

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "OpenVINO Clip requires open_clip to be installed.\n"
                "Please install transformers with "
                "`pip install open_clip_torch`."
            )

        preprocess = open_clip.image_transform(**processor_inputs)

        super().__init__(embed_batch_size=embed_batch_size, **kwargs)

        self._device = device
        self._visual_model = visual_model
        self._text_model = text_model
        self._tokenizer = tokenizer
        self._preprocess = preprocess

    # TEXT EMBEDDINGS

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=self._text_model.config.text_config.context_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        return self._text_model(tokens).text_features.tolist()

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    # IMAGE EMBEDDINGS

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        import torch

        with torch.no_grad():
            image = self._preprocess(Image.open(img_file_path)).unsqueeze(0)
            return self._visual_model(image).image_features.tolist()[0]
