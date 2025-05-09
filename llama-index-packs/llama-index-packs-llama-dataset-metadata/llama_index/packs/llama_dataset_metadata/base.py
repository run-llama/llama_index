import json
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.download.module import LLAMA_HUB_URL
from llama_index.core.download.utils import get_file_content
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.settings import Settings

if TYPE_CHECKING:
    from llama_index.core.llama_dataset import LabelledRagDataset


class Readme(BaseModel):
    """A simple class for creating a README.md string."""

    name: str
    _readme_template_path: str = "/llama_datasets/template_README.md"

    def _name_to_title_case(self) -> str:
        return " ".join(el.title() for el in self.name.split(" "))

    def _name_to_camel_case(self) -> str:
        return "".join(el.title() for el in self.name.split(" "))

    def _name_to_snake_case(self) -> str:
        return self.name.replace(" ", "_").lower()

    def _get_readme_str(self) -> str:
        text, _ = get_file_content(LLAMA_HUB_URL, self._readme_template_path)
        return text

    def create_readme(self) -> str:
        readme_str = self._get_readme_str()
        return readme_str.format(
            NAME=self._name_to_title_case(), NAME_CAMELCASE=self._name_to_camel_case()
        )


def to_camel(string: str) -> str:
    """Converts a given string to camel casing."""
    string_split = string.split("_")
    return string_split[0] + "".join(word.capitalize() for word in string_split[1:])


class BaseMetadata(BaseModel):
    """Base Metadata class."""

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class BaselineConfig(BaseMetadata):
    """Baseline config data class."""

    chunk_size: int
    llm: str
    similarity_top_k: int
    embed_model: str


class BaselineMetrics(BaseMetadata):
    """Baseline metrics data class."""

    context_similarity: Optional[float]
    correctness: float
    faithfulness: float
    relevancy: float


class Baseline(BaseMetadata):
    """Baseline data class."""

    name: str
    config: BaselineConfig
    metrics: BaselineMetrics
    code_url: str


class DatasetCard(BaseMetadata):
    """A pydantic BaseModel representing DatasetCard."""

    name: str
    class_name: str = "LabelledRagDataset"
    description: str
    number_observations: int
    contains_examples_by_humans: bool
    contains_examples_by_ai: bool
    source_urls: Optional[List[str]]
    baselines: List[Baseline]

    @staticmethod
    def _format_metric(val: float):
        """
        Formats a metric to 3 decimal places.

        Args:
            val (float): the value to format.

        """
        return float(f"{val:,.3f}")

    @classmethod
    def from_rag_evaluation(
        cls,
        index: BaseIndex,
        benchmark_df: pd.DataFrame,
        rag_dataset: "LabelledRagDataset",
        name: str,
        baseline_name: str,
        description: str,
        source_urls: Optional[List[str]] = None,
        code_url: Optional[str] = None,
    ) -> "DatasetCard":
        """
        Convenience constructor method for building a DatasetCard.

        Args:
            index (BaseIndex): the index from which query_engine is derived and
                used in the rag evaluation.
            benchmark_df (pd.DataFrame): the benchmark dataframe after using
                RagEvaluatorPack
            rag_dataset (LabelledRagDataset): the LabelledRagDataset used for
                evaluations
            name (str): The name of the new dataset e.g., "Paul Graham Essay Dataset"
            baseline_name (str): The name of the baseline e.g., "llamaindex"
            description (str): The description of the new dataset.
            source_urls (Optional[List[str]], optional): _description_. Defaults to None.
            code_url (Optional[str], optional): _description_. Defaults to None.

        Returns:
            DatasetCard

        """
        # extract metadata from rag_dataset
        num_observations = len(rag_dataset.examples)
        contains_examples_by_humans = any(
            (el.query_by.type == "human" or el.reference_answer_by.type == "human")
            for el in rag_dataset.examples
        )
        contains_examples_by_ai = any(
            (el.query_by.type == "ai" or el.reference_answer_by.type == "ai")
            for el in rag_dataset.examples
        )

        # extract baseline config info from index
        llm = Settings.llm.metadata.model_name
        embed_model = Settings.embed_model.model_name
        chunk_size = index._transformations[0].chunk_size
        similarity_top_k = index.as_retriever()._similarity_top_k
        baseline_config = BaselineConfig(
            llm=llm,
            chunk_size=chunk_size,
            similarity_top_k=similarity_top_k,
            embed_model=embed_model,
        )

        # extract baseline metrics from benchmark_df
        baseline_metrics = BaselineMetrics(
            correctness=cls._format_metric(
                benchmark_df.T["mean_correctness_score"].values[0]
            ),
            relevancy=cls._format_metric(
                benchmark_df.T["mean_relevancy_score"].values[0]
            ),
            faithfulness=cls._format_metric(
                benchmark_df.T["mean_faithfulness_score"].values[0]
            ),
            context_similarity=cls._format_metric(
                benchmark_df.T["mean_context_similarity_score"].values[0]
            ),
        )

        # baseline
        if code_url is None:
            code_url = ""
        baseline = Baseline(
            name=baseline_name,
            config=baseline_config,
            metrics=baseline_metrics,
            code_url=code_url,
        )

        if source_urls is None:
            source_urls = []

        return cls(
            name=name,
            description=description,
            source_urls=source_urls,
            number_observations=num_observations,
            contains_examples_by_humans=contains_examples_by_humans,
            contains_examples_by_ai=contains_examples_by_ai,
            baselines=[baseline],
        )


class LlamaDatasetMetadataPack(BaseLlamaPack):
    """
    A llamapack for creating and saving the necessary metadata files for
    submitting a llamadataset: card.json and README.md.
    """

    def run(
        self,
        index: BaseIndex,
        benchmark_df: pd.DataFrame,
        rag_dataset: "LabelledRagDataset",
        name: str,
        description: str,
        baseline_name: str,
        source_urls: Optional[List[str]] = None,
        code_url: Optional[str] = None,
    ):
        """
        Main usage for a llamapack. This will build the card.json and README.md
        and save them to local disk.

        Args:
            index (BaseIndex): the index from which query_engine is derived and
                used in the rag evaluation.
            benchmark_df (pd.DataFrame): the benchmark dataframe after using
                RagEvaluatorPack
            rag_dataset (LabelledRagDataset): the LabelledRagDataset used for
                evaluations
            name (str): The name of the new dataset e.g., "Paul Graham Essay Dataset"
            baseline_name (str): The name of the baseline e.g., "llamaindex"
            description (str): The description of the new dataset.
            source_urls (Optional[List[str]], optional): _description_. Defaults to None.
            code_url (Optional[str], optional): _description_. Defaults to None.

        """
        readme_obj = Readme(name=name)
        card_obj = DatasetCard.from_rag_evaluation(
            index=index,
            benchmark_df=benchmark_df,
            rag_dataset=rag_dataset,
            name=name,
            description=description,
            baseline_name=baseline_name,
            source_urls=source_urls,
            code_url=code_url,
        )

        # save card.json
        with open("card.json", "w") as f:
            json.dump(card_obj.dict(by_alias=True), f)

        # save README.md
        with open("README.md", "w") as f:
            f.write(readme_obj.create_readme())
