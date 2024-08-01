import asyncio
import math
import numpy as np
import random
import tqdm

from functools import reduce
from typing import Any, List, Dict, Sequence, Union, Coroutine, Iterable
from llama_index.core.async_utils import asyncio_module
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM, CompletionResponse
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
from llama_index.packs.diff_private_simple_dataset.events import (
    EmptyIntersectionEvent,
    LLMEmptyResponseEvent,
    SyntheticExampleEndEvent,
    SyntheticExampleStartEvent,
)
from prv_accountant.privacy_random_variables import (
    PoissonSubsampledGaussianMechanism,
    PureDPMechanism,
)
from prv_accountant import PRVAccountant

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


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


def _batch(iterable, n=1) -> Iterable[Any]:
    """Return iterable batches of an iterable."""
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class DiffPrivateSimpleDatasetPack(BaseLlamaPack):
    """A pack for creating differentially private simple llama-dataset."""

    def __init__(
        self,
        llm: LLM,  # currently only supports OpenAI completion LLMs
        tokenizer: Any,
        prompt_bundle: PromptBundle,
        simple_dataset: LabelledSimpleDataset,
        batch_size: int = 5,
        sleep_time_in_seconds: float = 0,
        sephamore_counter_size: int = 1,
        cache_checkpoints: bool = True,
        show_progress: bool = True,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_bundle = prompt_bundle
        self.simple_dataset = simple_dataset
        self._num_examples = len(self.simple_dataset.examples)
        self.labels = list({el.reference_label for el in self.simple_dataset[:]})
        self.sleep_time_in_seconds = sleep_time_in_seconds
        self._semaphore = asyncio.Semaphore(sephamore_counter_size)
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.cache_checkpoints = cache_checkpoints

    def sigma_to_eps(
        self,
        sigma: float,
        mechanism: PrivacyMechanism,
        size: int,
        max_token_cnt: int,
        max_self_compositions: int = 1000,
        eps_error: float = 0.01,
        delta_error: float = 1e-10,
    ) -> float:
        """Return the epsilon value given a sigma.

        Args:
            sigma (float): The parameter associated with noise mechanism.
            mechanism (PrivacyMechanism): Noise mechanism.
            size (int): Number of samples to be generated.
            max_token_cnt (int): Number of tokens generated per sample.
            max_self_compositions (int, optional): PRV algorithm parameter. Defaults to 1000.
            eps_error (float, optional): PRV algorithm parameter. Defaults to 0.01.
            delta_error (float, optional): PRV algorithm parameter. Defaults to 1e-10.

        Returns:
            float: The epsilon value.
        """
        if max_token_cnt > max_self_compositions:
            raise ValueError(
                "`max_token_cnt` cannot be greater than `max_self_composition`."
            )

        sample_rate = size / self._num_examples
        if mechanism == PrivacyMechanism.GAUSSIAN:
            prv_0 = PoissonSubsampledGaussianMechanism(
                noise_multiplier=sigma, sampling_probability=sample_rate
            )
        elif mechanism == PrivacyMechanism.EXPONENTIAL:
            sigma_bar = math.log(1 + sample_rate * (math.exp(sigma) - 1))
            prv_0 = PureDPMechanism(eps=sigma_bar)
        else:
            raise ValueError(
                "Invalid value for mechanism entered."
                " Please use either 'gaussian' or 'exponential'."
            )
        accountant = PRVAccountant(
            prvs=[
                prv_0,
            ],
            max_self_compositions=[max_self_compositions],
            eps_error=eps_error,
            delta_error=delta_error,
        )
        _eps_low, eps_est, _eps_up = accountant.compute_epsilon(
            delta=1 / self._num_examples, num_self_compositions=[max_token_cnt]
        )
        return eps_est

    async def _async_worker(self, job: Coroutine) -> Any:
        async with self._semaphore:
            await asyncio.sleep(self.sleep_time_in_seconds)
            return await job

    @dispatcher.span
    def _filter_dataset_by_label(self, label: str) -> LabelledSimpleDataset:
        """Filter simple_dataset by label."""
        if label not in self.labels:
            raise ValueError(
                "There are no examples with `label` in the associated `simple_dataset`."
            )
        examples = [el for el in self.simple_dataset[:] if el.reference_label == label]
        return LabelledSimpleDataset(examples=examples)

    @dispatcher.span
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
        self, split_probs: Dict[str, float], token_universe_proba: Dict[str, float]
    ) -> Dict[str, float]:
        """Normalize a probability distribution over tokens to become a valid probability distribution."""
        scale = sum(proba for proba in split_probs.values())
        if scale == 0:
            # universe
            dispatcher.event(
                EmptyIntersectionEvent(
                    public_tokens=list(token_universe_proba),
                    private_tokens=list(split_probs),
                )
            )
            split_probs = token_universe_proba  # use public probas instead
            scale = sum(proba for proba in split_probs.values())

        return {token: proba / scale for token, proba in split_probs.items()}

    def _extract_and_normalize_next_token_probas(
        self, response: CompletionResponse, token_universe_probas: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract and normalize LogProba from a CompletionResponse."""
        try:
            next_token_proba_distn = response.logprobs[0]
        except IndexError:
            dispatcher.event(LLMEmptyResponseEvent())
            return token_universe_probas
        except Exception as e:
            raise ValueError(
                "Something went wrong when trying to get LogProb from CompletionResponse."
            )

        split_probs = {t: 0 for t in token_universe_probas}
        for el in next_token_proba_distn:  # for immediate next token only
            if el.token in split_probs:
                split_probs[el.token] = np.exp(el.logprob)
        return self._normalize(
            split_probs, token_universe_probas
        )  # to make into a valid prob distribution

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

    @dispatcher.span
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

    @dispatcher.span
    async def agenerate_dp_synthetic_example(
        self,
        label: str,
        t_max: int = 1,
        sigma: float = 0.5,
        num_splits: int = 5,
        num_samples_per_split: int = 1,
    ) -> LabelledSimpleDataExample:
        """Generates a differentially private synthetic example."""
        dispatcher.event(SyntheticExampleStartEvent())
        synthetic_example = ""

        iterator = range(1, t_max + 1)
        if self.show_progress:
            iterator = tqdm.tqdm(iterator, position=0, leave=True)

        for _ in iterator:
            token_universe_prompt = self._get_public_prompt(
                synthetic_example=synthetic_example, label=label
            )
            try:
                response = await self._async_worker(
                    self.llm.acomplete(token_universe_prompt)
                )
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
            split_tasks = []
            for split in disjoint_splits:
                prompt = self._get_private_prompt(split, synthetic_example, label)
                split_tasks.append(self._async_worker(self.llm.acomplete(prompt)))

            split_responses: List[CompletionResponse] = await asyncio.gather(
                *split_tasks
            )

            # get and normalized next-token probas per split
            splits = [
                self._extract_and_normalize_next_token_probas(
                    response, token_universe_probas
                )
                for response in split_responses
            ]

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
            synthetic_example = synthetic_example.split("[RESULT]")[-1].strip()
        except Exception as e:
            synthetic_example = synthetic_example

        simple_example = LabelledSimpleDataExample(
            reference_label=label,
            text=synthetic_example,
            text_by=CreatedBy(type=CreatedByType.AI, model_name=self.llm.model),
        )
        dispatcher.event(SyntheticExampleEndEvent())
        return simple_example

    @dispatcher.span
    def run(
        self,
        sizes: Union[int, Dict[str, int]],
        t_max: int = 1,
        sigma: float = 0.5,
        num_splits: int = 5,
        num_samples_per_split: int = 1,
    ) -> LabelledSimpleDataset:
        """Main run method."""
        if num_samples_per_split < 1:
            raise ValueError(
                "`num_samples_per_split` must be an integer greater than 1."
            )

        if isinstance(sizes, int):
            sizes_dict = {d: sizes for d in self.labels}
        elif isinstance(sizes, dict):
            sizes_dict = sizes
        else:
            raise TypeError(
                "Invalid type of `sizes`. Must be either an `int` or `dict`."
            )

        if not all(c in sizes_dict for c in self.labels):
            raise ValueError("Not all labels have sizes.")

        examples = []
        for label in self.labels:
            size = sizes_dict[label]
            for _ in range(size):
                example = self.generate_dp_synthetic_example(
                    label=label,
                    t_max=t_max,
                    sigma=sigma,
                    num_splits=num_splits,
                    num_samples_per_split=num_samples_per_split,
                )
                examples.append(example)

        return LabelledSimpleDataset(examples=examples)

    @dispatcher.span
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

        if isinstance(sizes, int):
            sizes_dict = {d: sizes for d in self.labels}
        elif isinstance(sizes, dict):
            sizes_dict = sizes
        else:
            raise TypeError(
                "Invalid type of `sizes`. Must be either an `int` or `dict`."
            )

        if not all(c in sizes_dict for c in self.labels):
            raise ValueError("Not all labels have sizes.")

        tasks = []
        for label in self.labels:
            size = sizes_dict[label]
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

        # run in batch
        examples = []
        for batch in _batch(tasks, self.batch_size):
            batch_examples = await asyncio_runner.gather(*batch)
            examples += batch_examples
            if self.cache_checkpoints:
                checkpoint = LabelledSimpleDataset(examples=examples)
                checkpoint.save_json("checkpoint.json")

        return LabelledSimpleDataset(examples=examples)
