from typing import List, Dict
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llama_dataset.base import CreatedBy, CreatedByType
from llama_index.core.prompts.mixin import PromptMixin
from llama_index.packs.diff_private_simple_dataset.simple_dataset import (
    LabelledSimpleDataset,
    LabelledSimpleDataExample,
)
from llama_index.packs.diff_private_simple_dataset.templates import (
    zero_shot_chat_template,
    one_shot_chat_template,
    two_shot_chat_template,
)
from llama_index.packs.diff_private_simple_dataset.privacy_mechanism import (
    PrivacyMechanism,
)
import numpy as np
import random


class PromptBundle(BaseModel):
    instruction: str = Field(description="Instruction associated with underlying task.")
    text_heading: str = Field(description="Heading used for text.")
    label_heading: str = Field(description="Label heading used for label.")


class DiffPrivateSimpleDatasetPack(PromptMixin, BaseLlamaPack):
    """A pack for creating differentially private simple llama-dataset."""

    def __init__(
        self,
        llm: LLM,  # needs an LLM that produces logprobs one token at a time
        prompt_bundle: PromptBundle,
        simple_dataset: LabelledSimpleDataset,
        show_progress: bool = True,
    ):
        self.llm = llm
        self.prompt_bundle = prompt_bundle
        self.simple_dataset = simple_dataset
        self._num_examples = len(self.simple_dataset.examples)
        self._labels = list({el.reference_label for el in self.simple_dataset[:]})
        self.show_progress = show_progress
        self.prediction_dataset = None

    def _eps_to_sigma(eps: float, delta: float, mode: str = "gaussian") -> float:
        """Return the scale parameter with a given epsilon value.

        Source: https://programming-dp.com/ch6.html#the-gaussian-mechanism
        """
        sensitivity_upper_bound = np.sqrt(2)
        return (sensitivity_upper_bound * np.sqrt(np.log(1.25 / delta))) / eps

    def _filter_dataset_by_label(self, label: str) -> LabelledSimpleDataset:
        """Filter simple_dataset by label."""
        if label not in self._labels:
            raise ValueError(
                "There are no examples with `label` in the associated `simple_dataset`."
            )
        examples = [el for el in self.simple_dataset[:] if el.reference_label == label]
        return LabelledSimpleDataset(examples=examples)

    def _split_dataset(
        self,
        dataset: LabelledSimpleDataset,
        num_splits: int,
        num_samples_per_split: int,
    ) -> List[LabelledSimpleDataset]:
        """Splits a dataset into a set of disjoint datasets with equal number of examples."""
        indexes = list(range(len(dataset.examples)))
        random.shuffle(indexes)
        partitions = [indexes[i::num_splits] for i in range(num_splits)]
        splits = []
        for p in partitions:
            sample = random.sample(p, num_samples_per_split)
            if not len(sample) == num_samples_per_split:
                raise ValueError(
                    "Not able to create disjoint sets with current values of `num_splits` and `num_samples_per_split`."
                )
            examples = [dataset.examples[ix] for ix in sample]
            splits.append(LabelledSimpleDataset(examples=examples))
        return splits

    def _get_messages_for_reduced_token_universe(
        self,
        synthetic_example: str,
        label: str,
    ) -> List[ChatMessage]:
        """Get chat messages with instructions to produce the reduced next token universe."""
        return zero_shot_chat_template.format_messages(
            synthetic_text=synthetic_example,
            label=label,
            instruction=self.prompt_bundle.instruction,
            label_heading=self.prompt_bundle.label_heading,
            text_heading=self.prompt_bundle.text_heading,
        )

    def _get_messages_for_synthetic_generation(
        self,
        split: LabelledSimpleDataset,
        synthetic_example: str,
        label: str,
    ) -> List[ChatMessage]:
        """Get chat messages to produce the next token probabilities for a given split."""
        if len(split.examples) == 1:
            return one_shot_chat_template.format_messages(
                synthetic_text=synthetic_example,
                example_label=split.examples[0].label,
                example_text=split.examples[0].text,
                label=label,
                instruction=self.prompt_bundle.instruction,
                label_heading=self.prompt_bundle.label_heading,
                text_heading=self.prompt_bundle.text_heading,
            )
        else:
            return two_shot_chat_template.format_messages(
                synthetic_text=synthetic_example,
                example1_label=split.examples[0].label,
                example1_text=split.examples[0].text,
                example2_label=split.examples[0].label,
                example2_text=split.examples[0].text,
                label=label,
                instruction=self.prompt_bundle.instruction,
                label_heading=self.prompt_bundle.label_heading,
                text_heading=self.prompt_bundle.text_heading,
            )

    def _normalize(self, split_probs: Dict[str, float]) -> Dict[str, float]:
        """Normalize a probability distribution over tokens to become a valid probability distribution."""
        scale = sum(proba for proba in split_probs.values())
        return {token: proba / scale for token, proba in split_probs.items()}

    def _generate_noise(
        self, sigma: float, size: int, mechanism: PrivacyMechanism
    ) -> float:
        """Generates noise that satisfies eps-delta differential privacy."""
        noise_rng: np.random.RandomState
        if mechanism == PrivacyMechanism.GAUSSIAN:
            return noise_rng.normal(0, sigma, size=size)
        else:
            return 0.0  # not yet implemented

    def _merge_probas(self, list_of_probas: List[Dict[str, float]]) -> Dict[str, float]:
        """Merges a set of probabillity distributions over a common token universe."""
        scale = len(list_of_probas)
        tokens = list_of_probas[0].keys()
        merged_distribution = {}
        for token in tokens:
            merged_distribution[token] = sum(pr[token] / scale for pr in list_of_probas)
        return merged_distribution

    def _mode_of_distribution(self, proba: Dict[str, float]) -> str:
        """Returns the mode of a given probability distribution."""
        return max(proba, key=proba.get)

    def generate_dp_synthetic_example(
        self,
        label: str,
        t_max: int = 1,
        sigma: float = 0.5,
        num_splits: int = 5,
        num_samples_per_split: int = 1,
    ) -> LabelledSimpleDataExample:
        """Generates a differentially private synthetic example."""
        delta = 1 / self._num_examples
        synthetic_example = ""

        for _ in range(t_max):
            # reduced token universe
            token_universe_messages = self._get_messages_for_reduced_token_universe(
                synthetic_example
            )
            response = self.llm.chat(messages)
            token_universe_probas = {
                el.token: np.exp(el.logprob)
                for el in response.logprobs[0]  # only for next immediate token
            }

            # split the private dataset
            disjoint_splits = self._split_dataset(num_splits, num_samples_per_split)

            # generate next token probability distributions per split
            splits = []
            for split in disjoint_splits:
                split_probs = {token: 0 for token in token_universe_probas}
                messages = self._get_messages_for_synthetic_generation(
                    split, synthetic_example, label
                )
                response = self.llm.chat(messages)

                # updating and (rescaling) split probs
                for el in response.logprobs:
                    if el.token in split_probs:
                        split_probs[el.token] = np.exp(el.logprob)
                split_probs = self.normalize(
                    split_probs
                )  # to make into a valid prob distribution

                splits.append(split_probs)

            # noisy aggrergation
            sigma_calib = np.sqrt(2) / num_splits * sigma
            noise = self._generate_noise(sigma=sigma, size=len(token_universe_probas))
            agg_probs = self._merge_probas(splits) + noise

            # next token
            synthetic_example += self._mode_of_distribution(agg_probs)

        return LabelledSimpleDataExample(
            reference_label=label,
            text=synthetic_example,
            text_by=CreatedBy(type=CreatedByType.AI, model_name=self.llm.model),
        )

    def run(
        self,
        sizes: Dict[str, int],
        t_max: int = 1,
        sigma: float = 0.5,
        num_splits: int = 5,
        num_samples_per_split: int = 1,
    ) -> LabelledSimpleDataset:
        """Main run method."""
        if num_samples_per_split not in [1, 2]:
            raise ValueError("`num_samples_per_split` can only be either 1 or 2.")

        if not all(c is sizes.keys() for c in self._labels):
            raise ValueError("Not all labels have sizes.")

        examples = []
        for label in self._labels:
            size = sizes[label]
            for _ in range(size):
                example = self.generate_dp_synthetic_example(
                    label=label,
                    t_max=t_max,
                    sigma=sigma,
                    num_splits=num_splits,
                    num_samples_per_split=num_samples_per_split,
                )
                examples.append(example)

        return LabelledSimpleDataset(examples=example)
