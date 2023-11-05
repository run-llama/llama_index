"""Param tuner."""


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from copy import deepcopy
from typing import Callable
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
    param_fn: Callable[[Dict[str, Any]], RunResult] = Field(
        ..., description="Function to run with parameters."
    )
    param_dict: Dict[str, Any] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    fixed_param_dict: Dict[str, Any] = Field(
        default_factory=dict, description="A dictionary of fixed parameters passed to each job."
    )
    show_progress: bool = False

    @abstractmethod
    def tune(self) -> TunedResult:
        """Tune parameters."""


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

    # def __init__(
    #     self, 
    #     param_fn: Callable[[Dict[str, Any]], RunResult],
    #     param_dict: Dict, 
    #     fixed_param_dict: Dict,
    #     show_progress: bool = False
    # ) -> None:
    #     """Init params."""
    #     self._param_dict = param_dict
    #     self._fixed_param_dict = fixed_param_dict
    #     self._param_fn = param_fn
    #     self._show_progress = show_progress

    def tune(self) -> TunedResult:
        """Run tuning."""
        # each key in param_dict is a parameter to tune, each val
        # is a list of values to try
        # generate combinations of parameters from the param_dict
        param_combinations = generate_param_combinations(self.param_dict)

        # for each combination, run the job with the arguments
        # in args_dict

        combos_with_progress = enumerate(
            get_tqdm_iterable(param_combinations, self.show_progress, "Param combinations.")
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

        def param_fn_wrapper(ray_param_dict: Dict, fixed_param_dict: Optional[Dict] = None):
            # need a wrapper to pass in parameters to tune + fixed params
            fixed_param_dict = fixed_param_dict or {}
            full_param_dict = {
                **fixed_param_dict,
                **ray_param_dict,
            }
            tuned_result = self.param_fn(full_param_dict)
            return tuned_result.dict()

        run_config = RunConfig(**self.run_config_dict) if self.run_config_dict else None

        tuner = tune.Tuner(
            tune.with_parameters(param_fn_wrapper, fixed_param_dict=self.fixed_param_dict),
            param_space=ray_param_dict,
            run_config=run_config,
        )

        results = tuner.fit()
        all_run_results = []
        for result in results:
            # convert dict back to RunResult (reconstruct it with metadata)
            # get the keys in RunResult, assign corresponding values in 
            # result.metrics to those keys
            run_result = RunResult.parse_obj(result.metrics)
            # add some more metadata to run_result (e.g. timestamp)
            run_result.metadata["timestamp"] = result.metrics["timestamp"]

            all_run_results.append(run_result)

        # sort the results by score
        sorted_run_results = sorted(
            all_run_results, key=lambda x: x.score, reverse=True
        )

        return TunedResult(run_results=sorted_run_results, best_idx=0)
