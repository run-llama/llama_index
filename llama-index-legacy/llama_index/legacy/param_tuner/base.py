"""Param tuner."""


import asyncio
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional

from llama_index.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.utils import get_tqdm_iterable


class RunResult(BaseModel):
    """Run result."""

    score: float
    params: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata.")


class TunedResult(BaseModel):
    run_results: List[RunResult]
    best_idx: int

    @property
    def best_run_result(self) -> RunResult:
        """Get best run result."""
        return self.run_results[self.best_idx]


def generate_param_combinations(param_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate parameter combinations."""

    def _generate_param_combinations_helper(
        param_dict: Dict[str, Any], curr_param_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Helper function."""
        if len(param_dict) == 0:
            return [deepcopy(curr_param_dict)]
        param_dict = deepcopy(param_dict)
        param_name, param_vals = param_dict.popitem()
        param_combinations = []
        for param_val in param_vals:
            curr_param_dict[param_name] = param_val
            param_combinations.extend(
                _generate_param_combinations_helper(param_dict, curr_param_dict)
            )
        return param_combinations

    return _generate_param_combinations_helper(param_dict, {})


class BaseParamTuner(BaseModel):
    """Base param tuner."""

    param_dict: Dict[str, Any] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    fixed_param_dict: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of fixed parameters passed to each job.",
    )
    show_progress: bool = False

    @abstractmethod
    def tune(self) -> TunedResult:
        """Tune parameters."""

    async def atune(self) -> TunedResult:
        """Async Tune parameters.

        Override if you implement a native async method.

        """
        return self.tune()


class ParamTuner(BaseParamTuner):
    """Parameter tuner.

    Args:
        param_dict(Dict): A dictionary of parameters to iterate over.
            Example param_dict:
            {
                "num_epochs": [10, 20],
                "batch_size": [8, 16, 32],
            }
        fixed_param_dict(Dict): A dictionary of fixed parameters passed to each job.

    """

    param_fn: Callable[[Dict[str, Any]], RunResult] = Field(
        ..., description="Function to run with parameters."
    )

    def tune(self) -> TunedResult:
        """Run tuning."""
        # each key in param_dict is a parameter to tune, each val
        # is a list of values to try
        # generate combinations of parameters from the param_dict
        param_combinations = generate_param_combinations(self.param_dict)

        # for each combination, run the job with the arguments
        # in args_dict

        combos_with_progress = enumerate(
            get_tqdm_iterable(
                param_combinations, self.show_progress, "Param combinations."
            )
        )

        all_run_results = []
        for idx, param_combination in combos_with_progress:
            full_param_dict = {
                **self.fixed_param_dict,
                **param_combination,
            }
            run_result = self.param_fn(full_param_dict)

            all_run_results.append(run_result)

        # sort the results by score
        sorted_run_results = sorted(
            all_run_results, key=lambda x: x.score, reverse=True
        )

        return TunedResult(run_results=sorted_run_results, best_idx=0)


class AsyncParamTuner(BaseParamTuner):
    """Async Parameter tuner.

    Args:
        param_dict(Dict): A dictionary of parameters to iterate over.
            Example param_dict:
            {
                "num_epochs": [10, 20],
                "batch_size": [8, 16, 32],
            }
        fixed_param_dict(Dict): A dictionary of fixed parameters passed to each job.
        aparam_fn (Callable): An async function to run with parameters.
        num_workers (int): Number of workers to use.

    """

    aparam_fn: Callable[[Dict[str, Any]], Awaitable[RunResult]] = Field(
        ..., description="Async function to run with parameters."
    )
    num_workers: int = Field(2, description="Number of workers to use.")

    _semaphore: asyncio.Semaphore = PrivateAttr()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._semaphore = asyncio.Semaphore(self.num_workers)

    async def atune(self) -> TunedResult:
        """Run tuning."""
        # each key in param_dict is a parameter to tune, each val
        # is a list of values to try
        # generate combinations of parameters from the param_dict
        param_combinations = generate_param_combinations(self.param_dict)

        # for each combination, run the job with the arguments
        # in args_dict

        async def aparam_fn_worker(
            semaphore: asyncio.Semaphore,
            full_param_dict: Dict[str, Any],
        ) -> RunResult:
            """Async param fn worker."""
            async with semaphore:
                return await self.aparam_fn(full_param_dict)

        all_run_results = []
        run_jobs = []
        for param_combination in param_combinations:
            full_param_dict = {
                **self.fixed_param_dict,
                **param_combination,
            }
            run_jobs.append(aparam_fn_worker(self._semaphore, full_param_dict))
            # run_jobs.append(self.aparam_fn(full_param_dict))

        if self.show_progress:
            from tqdm.asyncio import tqdm_asyncio

            all_run_results = await tqdm_asyncio.gather(*run_jobs)
        else:
            all_run_results = await asyncio.gather(*run_jobs)

        # sort the results by score
        sorted_run_results = sorted(
            all_run_results, key=lambda x: x.score, reverse=True
        )

        return TunedResult(run_results=sorted_run_results, best_idx=0)

    def tune(self) -> TunedResult:
        """Run tuning."""
        return asyncio.run(self.atune())


class RayTuneParamTuner(BaseParamTuner):
    """Parameter tuner powered by Ray Tune.

    Args:
        param_dict(Dict): A dictionary of parameters to iterate over.
            Example param_dict:
            {
                "num_epochs": [10, 20],
                "batch_size": [8, 16, 32],
            }
        fixed_param_dict(Dict): A dictionary of fixed parameters passed to each job.

    """

    param_fn: Callable[[Dict[str, Any]], RunResult] = Field(
        ..., description="Function to run with parameters."
    )

    run_config_dict: Optional[dict] = Field(
        default=None, description="Run config dict for Ray Tune."
    )

    def tune(self) -> TunedResult:
        """Run tuning."""
        from ray import tune
        from ray.train import RunConfig

        # convert every array in param_dict to a tune.grid_search
        ray_param_dict = {}
        for param_name, param_vals in self.param_dict.items():
            ray_param_dict[param_name] = tune.grid_search(param_vals)

        def param_fn_wrapper(
            ray_param_dict: Dict, fixed_param_dict: Optional[Dict] = None
        ) -> Dict:
            # need a wrapper to pass in parameters to tune + fixed params
            fixed_param_dict = fixed_param_dict or {}
            full_param_dict = {
                **fixed_param_dict,
                **ray_param_dict,
            }
            tuned_result = self.param_fn(full_param_dict)
            # need to convert RunResult to dict to obey
            # Ray Tune's API
            return tuned_result.dict()

        run_config = RunConfig(**self.run_config_dict) if self.run_config_dict else None

        tuner = tune.Tuner(
            tune.with_parameters(
                param_fn_wrapper, fixed_param_dict=self.fixed_param_dict
            ),
            param_space=ray_param_dict,
            run_config=run_config,
        )

        results = tuner.fit()
        all_run_results = []
        for idx in range(len(results)):
            result = results[idx]
            # convert dict back to RunResult (reconstruct it with metadata)
            # get the keys in RunResult, assign corresponding values in
            # result.metrics to those keys
            run_result = RunResult.parse_obj(result.metrics)
            # add some more metadata to run_result (e.g. timestamp)
            run_result.metadata["timestamp"] = (
                result.metrics["timestamp"] if result.metrics else None
            )

            all_run_results.append(run_result)

        # sort the results by score
        sorted_run_results = sorted(
            all_run_results, key=lambda x: x.score, reverse=True
        )

        return TunedResult(run_results=sorted_run_results, best_idx=0)
