import asyncio
from typing import Any, List, Dict, Sequence
from llama_index.core.async_utils import asyncio_module
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llama_dataset.base import CreatedBy, CreatedByType
from llama_index.core.llama_dataset.simple import (
    LabelledSimpleDataset,
    LabelledSimpleDataExample,
)
from llama_index.packs.diff_private_simple_dataset.templates import (
    zero_shot_completion_template,
    few_shot_completion_template,
    single_example_template,
)
from llama_index.packs.diff_private_simple_dataset.privacy_mechanism import (
    PrivacyMechanism,
)
from functools import reduce
import numpy as np
import random
import tqdm


STOP_TOKENS = {"<|endoftext|>", " END", "<|end|>"}

FALLBACK_SYNTHETIC_EXAMPLE = LabelledSimpleDataExample(
    reference_label="FALLBACK",
    text="DO NOT USE.",
    text_by=CreatedBy(type=CreatedByType.HUMAN),
)


class PromptBundle(BaseModel):
    instruction: str = Field(description="Instruction associated with underlying task.")
    text_heading: str = Field(description="Heading used for text.")
    label_heading: str = Field(description="Label heading used for label.")


class DiffPrivateSimpleDatasetPack(BaseLlamaPack):
    """A pack for creating differentially private simple llama-dataset."""

    def __init__(
        self,
        llm: LLM,  # currently only supports OpenAI completion LLMs
        tokenizer: Any,
        prompt_bundle: PromptBundle,
        simple_dataset: LabelledSimpleDataset,
        show_progress: bool = True,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_bundle = prompt_bundle
        self.simple_dataset = simple_dataset
        self._num_examples = len(self.simple_dataset.examples)
        self._labels = list({el.reference_label for el in self.simple_dataset[:]})
        self.show_progress = show_progress

    @staticmethod
    def eps_to_sigma(eps: float, delta: float, mode: str = "gaussian") -> float:
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

    def _get_public_prompt(
        self,
        synthetic_example: str,
        label: str,
    ) -> str:
        """Get completion prompt for token universe."""
        return zero_shot_completion_template.format(
            synthetic_text=synthetic_example,
            label=label,
            instruction=self.prompt_bundle.instruction,
            label_heading=self.prompt_bundle.label_heading,
            text_heading=self.prompt_bundle.text_heading,
        )

    def _get_private_prompt(
        self,
        split: LabelledSimpleDataset,
        synthetic_example: str,
        label: str,
    ) -> str:
        """Get prompt for completion endpoint."""
        single_templates = [
            single_example_template.format(
                label_heading=self.prompt_bundle.label_heading,
                text_heading=self.prompt_bundle.text_heading,
                example_label=x.reference_label,
                example_text=x.text,
            )
            for x in split.examples
        ]

        few_shot_examples = reduce(lambda x, y: x + y, single_templates)
        return few_shot_completion_template.format(
            instruction=self.prompt_bundle.instruction,
            label_heading=self.prompt_bundle.label_heading,
            text_heading=self.prompt_bundle.text_heading,
            few_shot_examples=few_shot_examples,
            label=label,
            synthetic_text=synthetic_example,
        )

    def _normalize(
        self, split_probs: Dict[str, float], universe_proba: Dict[str, float]
    ) -> Dict[str, float]:
        """Normalize a probability distribution over tokens to become a valid probability distribution."""
        scale = sum(proba for proba in split_probs.values())
        if scale == 0:
            # universe
            return universe_proba
        return {token: proba / scale for token, proba in split_probs.items()}

    def _generate_noise(
        self, sigma: float, size: int, mechanism: PrivacyMechanism
    ) -> float:
        """Generates noise that satisfies eps-delta differential privacy."""
        noise_rng = np.random.RandomState()
        if mechanism == PrivacyMechanism.GAUSSIAN:
            return noise_rng.normal(0, sigma, size=size)
        elif mechanism == PrivacyMechanism.LAPLACE:
            return noise_rng.exponential(scale=sigma, size=size)
        else:
            raise ValueError("Value entered for `mechanism` is not supported.")

    def _merge_probas(self, list_of_probas: List[Dict[str, float]]) -> Dict[str, float]:
        """Merges a set of probabillity distributions over a common token universe."""
        scale = len(list_of_probas)
        tokens = list_of_probas[0].keys()
        merged_distribution = {}
        for token in tokens:
            merged_distribution[token] = sum(pr[token] / scale for pr in list_of_probas)
        return merged_distribution

    def _add_noise(
        self, proba: Dict[str, float], noise_array=Sequence[float]
    ) -> Dict[str, float]:
        """Add noise to proba distribution."""
        return {
            token: proba + noise
            for (token, proba), noise in zip(proba.items(), noise_array)
        }

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
        return asyncio.run(
            self.agenerate_dp_synthetic_example(
                label=label,
                t_max=t_max,
                sigma=sigma,
                num_splits=num_splits,
                num_samples_per_split=num_samples_per_split,
            )
        )

    async def agenerate_dp_synthetic_example(
        self,
        label: str,
        t_max: int = 1,
        sigma: float = 0.5,
        num_splits: int = 5,
        num_samples_per_split: int = 1,
    ) -> LabelledSimpleDataExample:
        """Generates a differentially private synthetic example."""
        synthetic_example = ""

        iterator = range(1, t_max + 1)
        if self.show_progress:
            iterator = tqdm.tqdm(iterator)

        for _ in range(1, t_max + 1):
            token_universe_prompt = self._get_public_prompt(
                synthetic_example=synthetic_example, label=label
            )
            try:
                response = await self.llm.acomplete(token_universe_prompt)
                token_universe_probas = {
                    el.token: np.exp(el.logprob)
                    for el in response.logprobs[0]  # only for next immediate token
                }
            except IndexError as e:
                continue  # try again in next iteration

            # filter dataset by label
            filtered_simple_dataset = self._filter_dataset_by_label(label=label)

            # split the private dataset
            disjoint_splits = self._split_dataset(
                dataset=filtered_simple_dataset,
                num_splits=num_splits,
                num_samples_per_split=num_samples_per_split,
            )

            # generate next token probability distributions per split
            splits = []
            for split in disjoint_splits:
                split_probs = {token: 0 for token in token_universe_probas}
                prompt = self._get_private_prompt(split, synthetic_example, label)
                response = self.llm.complete(prompt)
                # updating and (rescaling) split probs
                try:
                    next_token_proba_distn = response.logprobs[0]
                except IndexError:
                    continue  # move on to next iteration to try again

                for el in next_token_proba_distn:  # for immediate next token only
                    if el.token in split_probs:
                        split_probs[el.token] = np.exp(el.logprob)
                split_probs = self._normalize(
                    split_probs, token_universe_probas
                )  # to make into a valid prob distribution
                splits.append(split_probs)

            # noisy aggrergation
            sigma_calib = np.sqrt(2) / num_splits * sigma
            noise_array = self._generate_noise(
                sigma=sigma_calib, size=len(token_universe_probas), mechanism="gaussian"
            )
            merged_probas = self._merge_probas(splits)
            noisy_probs = self._add_noise(merged_probas, noise_array)

            # next token
            next_token = self._mode_of_distribution(noisy_probs)
            if next_token in STOP_TOKENS:
                break
            else:
                synthetic_example += next_token

        # synthetic example remove [RESULT]
        try:
            synthetic_example = synthetic_example.split("[RESULT] ")[-1]
        except Exception as e:
            synthetic_example = synthetic_example

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
        return asyncio.run(
            self.run(
                sizes=sizes,
                t_max=t_max,
                sigma=sigma,
                num_splits=num_splits,
                num_samples_per_split=num_samples_per_split,
            )
        )

    async def arun(
        self,
        sizes: Dict[str, int],
        t_max: int = 1,
        sigma: float = 0.5,
        num_splits: int = 5,
        num_samples_per_split: int = 1,
    ) -> LabelledSimpleDataset:
        """Main async run method."""
        if num_samples_per_split < 1:
            raise ValueError(
                "`num_samples_per_split` must be an integer greater than 1."
            )

        if not all(c in sizes for c in self._labels):
            raise ValueError("Not all labels have sizes.")

        tasks = []
        for label in self._labels:
            size = sizes[label]
            for _ in range(size):
                example_task = self.agenerate_dp_synthetic_example(
                    label=label,
                    t_max=t_max,
                    sigma=sigma,
                    num_splits=num_splits,
                    num_samples_per_split=num_samples_per_split,
                )
                tasks.append(example_task)

        asyncio_runner = asyncio_module(self.show_progress)

        examples = await asyncio_runner.gather(*tasks)

        return LabelledSimpleDataset(examples=examples)
