from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode
from llama_index.core.utils import get_tqdm_iterable
from nltk.tokenize import word_tokenize
from span_marker import SpanMarkerModel

DEFAULT_ENTITY_MAP = {
    "PER": "persons",
    "ORG": "organizations",
    "LOC": "locations",
    "ANIM": "animals",
    "BIO": "biological",
    "CEL": "celestial",
    "DIS": "diseases",
    "EVE": "events",
    "FOOD": "foods",
    "INST": "instruments",
    "MEDIA": "media",
    "PLANT": "plants",
    "MYTH": "mythological",
    "TIME": "times",
    "VEHI": "vehicles",
}

DEFAULT_ENTITY_MODEL = "tomaarsen/span-marker-mbert-base-multinerd"


class EntityExtractor(BaseExtractor):
    """
    Entity extractor. Extracts `entities` into a metadata field using a default model
    `tomaarsen/span-marker-mbert-base-multinerd` and the SpanMarker library.

    Install SpanMarker with `pip install span-marker`.
    """

    model_name: str = Field(
        default=DEFAULT_ENTITY_MODEL,
        description="The model name of the SpanMarker model to use.",
    )
    prediction_threshold: float = Field(
        default=0.5,
        description="The confidence threshold for accepting predictions.",
        ge=0.0,
        le=1.0,
    )
    span_joiner: str = Field(
        default=" ", description="The separator between entity names."
    )
    label_entities: bool = Field(
        default=False, description="Include entity class labels or not."
    )
    device: Optional[str] = Field(
        default=None, description="Device to run model on, i.e. 'cuda', 'cpu'"
    )
    entity_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of entity class names to usable names.",
    )

    _tokenizer: Callable = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_ENTITY_MODEL,
        prediction_threshold: float = 0.5,
        span_joiner: str = " ",
        label_entities: bool = False,
        device: Optional[str] = None,
        entity_map: Optional[Dict[str, str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        **kwargs: Any,
    ):
        """
        Entity extractor for extracting entities from text and inserting
        into node metadata.

        Args:
            model_name (str):
                Name of the SpanMarker model to use.
            prediction_threshold (float):
                Minimum prediction threshold for entities. Defaults to 0.5.
            span_joiner (str):
                String to join spans with. Defaults to " ".
            label_entities (bool):
                Whether to label entities with their type. Setting to true can be
                slightly error prone, but can be useful for downstream tasks.
                Defaults to False.
            device (Optional[str]):
                Device to use for SpanMarker model, i.e. "cpu" or "cuda".
                Loads onto "cpu" by default.
            entity_map (Optional[Dict[str, str]]):
                Mapping from entity class name to label.
            tokenizer (Optional[Callable[[str], List[str]]]):
                Tokenizer to use for splitting text into words.
                Defaults to NLTK word_tokenize.

        """
        base_entity_map = DEFAULT_ENTITY_MAP
        if entity_map is not None:
            base_entity_map.update(entity_map)

        super().__init__(
            model_name=model_name,
            prediction_threshold=prediction_threshold,
            span_joiner=span_joiner,
            label_entities=label_entities,
            device=device,
            entity_map=base_entity_map,
            **kwargs,
        )

        self._model = SpanMarkerModel.from_pretrained(model_name)
        if device is not None:
            self._model = self._model.to(device)

        self._tokenizer = tokenizer or word_tokenize

    @classmethod
    def class_name(cls) -> str:
        return "EntityExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        # Extract node-level entity metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        metadata_queue: Iterable[int] = get_tqdm_iterable(
            range(len(nodes)), self.show_progress, "Extracting entities"
        )

        for i in metadata_queue:
            metadata = metadata_list[i]
            node_text = nodes[i].get_content(metadata_mode=self.metadata_mode)
            words = self._tokenizer(node_text)
            spans = self._model.predict(words)
            for span in spans:
                if span["score"] > self.prediction_threshold:
                    ent_label = self.entity_map.get(span["label"], span["label"])
                    metadata_label = ent_label if self.label_entities else "entities"

                    if metadata_label not in metadata:
                        metadata[metadata_label] = set()

                    metadata[metadata_label].add(self.span_joiner.join(span["span"]))

        # convert metadata from set to list
        for metadata in metadata_list:
            for key, val in metadata.items():
                metadata[key] = list(val)

        return metadata_list
