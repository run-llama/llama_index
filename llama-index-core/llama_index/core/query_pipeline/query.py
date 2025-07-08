"""Query Pipeline."""

import deprecated
import json
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    get_args,
)

import networkx

from llama_index.core.async_utils import asyncio_run, run_jobs
from llama_index.core.bridge.pydantic import Field, ConfigDict
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base.query_pipeline.query import (
    QUERY_COMPONENT_TYPE,
    ChainableMixin,
    InputKeys,
    Link,
    OutputKeys,
    QueryComponent,
    ComponentIntermediates,
)
from llama_index.core.utils import print_text
from llama_index.core.query_pipeline.components.stateful import BaseStatefulComponent
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


# TODO: Make this (safely) pydantic?
class RunState:
    def __init__(
        self,
        module_dict: Dict[str, QueryComponent],
        module_input_dict: Dict[str, Dict[str, Any]],
    ):
        self.all_module_inputs: Dict[str, Dict[str, Any]] = {
            module_key: {} for module_key in module_dict
        }

        for module_key, input_dict in module_input_dict.items():
            self.all_module_inputs[module_key] = input_dict

        self.module_dict = module_dict
        self.result_outputs: Dict[str, Any] = {}
        self.intermediate_outputs: Dict[str, ComponentIntermediates] = {}
        self.executed_modules: Set[str] = set()


def get_output(
    src_key: Optional[str],
    output_dict: Dict[str, Any],
) -> Any:
    """Add input to module deps inputs."""
    # get relevant output from link
    if src_key is None:
        # ensure that output_dict only has one key
        if len(output_dict) != 1:
            raise ValueError("Output dict must have exactly one key.")
        output = next(iter(output_dict.values()))
    else:
        output = output_dict[src_key]
    return output


def add_output_to_module_inputs(
    dest_key: str,
    output: Any,
    module: QueryComponent,
    module_inputs: Dict[str, Any],
) -> None:
    """Add input to module deps inputs."""
    # now attach output to relevant input key for module
    if dest_key is None:
        free_keys = module.free_req_input_keys
        # ensure that there is only one remaining key given partials
        if len(free_keys) != 1:
            raise ValueError(
                "Module input keys must have exactly one key if "
                "dest_key is not specified. Remaining keys: "
                f"in module: {free_keys}"
            )
        module_inputs[next(iter(free_keys))] = output
    else:
        module_inputs[dest_key] = output


def print_debug_input(
    module_key: str,
    input: Dict[str, Any],
    val_str_len: int = 200,
) -> None:
    """Print debug input."""
    output = f"> Running module {module_key} with input: \n"
    for key, value in input.items():
        # stringify and truncate output
        val_str = (
            str(value)[:val_str_len] + "..."
            if len(str(value)) > val_str_len
            else str(value)
        )
        output += f"{key}: {val_str}\n"

    print_text(output + "\n", color="llama_lavender")


def print_debug_input_multi(
    module_keys: List[str],
    module_inputs: List[Dict[str, Any]],
    val_str_len: int = 200,
) -> None:
    """Print debug input."""
    output = f"> Running modules and inputs in parallel: \n"
    for module_key, input in zip(module_keys, module_inputs):
        cur_output = f"Module key: {module_key}. Input: \n"
        for key, value in input.items():
            # stringify and truncate output
            val_str = (
                str(value)[:val_str_len] + "..."
                if len(str(value)) > val_str_len
                else str(value)
            )
            cur_output += f"{key}: {val_str}\n"
        output += cur_output + "\n"

    print_text(output + "\n", color="llama_lavender")


# Function to clean non-serializable attributes and return a copy of the graph
# https://stackoverflow.com/questions/23268421/networkx-how-to-access-attributes-of-objects-as-nodes
def clean_graph_attributes_copy(graph: networkx.MultiDiGraph) -> networkx.MultiDiGraph:
    # Create a deep copy of the graph to preserve the original
    graph_copy = graph.copy()

    # Iterate over nodes and clean attributes
    for node, attributes in graph_copy.nodes(data=True):
        for key, value in list(attributes.items()):
            if callable(value):  # Checks if the value is a function
                del attributes[key]  # Remove the attribute if it's non-serializable

    # Similarly, you can extend this to clean edge attributes if necessary
    for u, v, attributes in graph_copy.edges(data=True):
        for key, value in list(attributes.items()):
            if callable(value):  # Checks if the value is a function
                del attributes[key]  # Remove the attribute if it's non-serializable

    return graph_copy


def get_stateful_components(
    query_component: QueryComponent,
) -> List[BaseStatefulComponent]:
    """Get stateful components."""
    stateful_components: List[BaseStatefulComponent] = []
    for c in query_component.sub_query_components:
        if isinstance(c, BaseStatefulComponent):
            stateful_components.append(cast(BaseStatefulComponent, c))

        if len(c.sub_query_components) > 0:
            stateful_components.extend(get_stateful_components(c))

    return stateful_components


def update_stateful_components(
    stateful_components: List[BaseStatefulComponent], state: Dict[str, Any]
) -> None:
    """Update stateful components."""
    for stateful_component in stateful_components:
        # stateful_component.partial(state=state)
        stateful_component.state = state


def get_and_update_stateful_components(
    query_component: QueryComponent, state: Dict[str, Any]
) -> List[BaseStatefulComponent]:
    """
    Get and update stateful components.

    Assign all stateful components in the query component with the state.

    """
    stateful_components = get_stateful_components(query_component)
    update_stateful_components(stateful_components, state)
    return stateful_components


CHAIN_COMPONENT_TYPE = Union[QUERY_COMPONENT_TYPE, str]


@deprecated.deprecated(
    reason=(
        "QueryPipeline has been deprecated and is not maintained.\n\n"
        "This implementation will be removed in a v0.13.0.\n\n"
        "It is recommended to switch to the Workflows API for a more flexible and powerful experience.\n\n"
        "See the docs for more information workflows: https://docs.llamaindex.ai/en/stable/understanding/workflows/"
    ),
    action="once",
)
class QueryPipeline(QueryComponent):
    """
    A query pipeline that can allow arbitrary chaining of different modules.

    A pipeline itself is a query component, and can be used as a module in another pipeline.

    DEPRECATED: QueryPipeline has been deprecated and is not maintained.
    This implementation will be removed in a v0.13.0.
    It is recommended to switch to the Workflows API for a more flexible and powerful experience.
    See the docs for more information workflows: https://docs.llamaindex.ai/en/stable/understanding/workflows/
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )

    module_dict: Dict[str, QueryComponent] = Field(
        default_factory=dict, description="The modules in the pipeline."
    )
    dag: networkx.MultiDiGraph = Field(
        default_factory=networkx.MultiDiGraph, description="The DAG of the pipeline."
    )
    verbose: bool = Field(
        default=False, description="Whether to print intermediate steps."
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show progress bar (currently async only).",
    )
    num_workers: int = Field(
        default=4, description="Number of workers to use (currently async only)."
    )
    state: Dict[str, Any] = Field(
        default_factory=dict, description="State of the pipeline."
    )

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        chain: Optional[Sequence[CHAIN_COMPONENT_TYPE]] = None,
        modules: Optional[Dict[str, QUERY_COMPONENT_TYPE]] = None,
        links: Optional[List[Link]] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        state = state or {}
        super().__init__(
            callback_manager=callback_manager or CallbackManager([]),
            state=state,
            **kwargs,
        )

        self._init_graph(chain=chain, modules=modules, links=links)
        # Pydantic validator isn't called for __init__ so we need to call it manually
        get_and_update_stateful_components(self, state)

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state."""
        self.state = state
        get_and_update_stateful_components(self, state)

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update state."""
        self.state.update(state)
        get_and_update_stateful_components(self, state)

    def reset_state(self) -> None:
        """Reset state."""
        # use pydantic validator to update state
        self.set_state({})

    def _init_graph(
        self,
        chain: Optional[Sequence[CHAIN_COMPONENT_TYPE]] = None,
        modules: Optional[Dict[str, QUERY_COMPONENT_TYPE]] = None,
        links: Optional[List[Link]] = None,
    ) -> None:
        """Initialize graph."""
        if chain is not None:
            if modules is not None or links is not None:
                raise ValueError("Cannot specify both chain and modules/links in init.")
            self.add_chain(chain)
        elif modules is not None:
            self.add_modules(modules)
            if links is not None:
                for link in links:
                    self.add_link(**link.model_dump())

    def add_chain(self, chain: Sequence[CHAIN_COMPONENT_TYPE]) -> None:
        """
        Add a chain of modules to the pipeline.

        This is a special form of pipeline that is purely sequential/linear.
        This allows a more concise way of specifying a pipeline.

        """
        # first add all modules
        module_keys: List[str] = []
        for module in chain:
            if isinstance(module, get_args(QUERY_COMPONENT_TYPE)):
                module_key = str(uuid.uuid4())
                self.add(module_key, cast(QUERY_COMPONENT_TYPE, module))
                module_keys.append(module_key)
            elif isinstance(module, str):
                module_keys.append(module)
            else:
                raise ValueError("Chain must be a sequence of modules or module keys.")

        # then add all links
        for i in range(len(chain) - 1):
            self.add_link(src=module_keys[i], dest=module_keys[i + 1])

    @property
    def stateful_components(self) -> List[BaseStatefulComponent]:
        """Get stateful component."""
        return get_stateful_components(self)

    def add_links(
        self,
        links: List[Link],
    ) -> None:
        """Add links to the pipeline."""
        for link in links:
            if isinstance(link, Link):
                self.add_link(**link.model_dump())
            else:
                raise ValueError("Link must be of type `Link` or `ConditionalLinks`.")

    def add_modules(self, module_dict: Dict[str, QUERY_COMPONENT_TYPE]) -> None:
        """Add modules to the pipeline."""
        for module_key, module in module_dict.items():
            self.add(module_key, module)

    def add(self, module_key: str, module: QUERY_COMPONENT_TYPE) -> None:
        """Add a module to the pipeline."""
        # if already exists, raise error
        if module_key in self.module_dict:
            raise ValueError(f"Module {module_key} already exists in pipeline.")

        if isinstance(module, ChainableMixin):
            module = module.as_query_component()
        else:
            pass

        self.module_dict[module_key] = cast(QueryComponent, module)
        self.dag.add_node(module_key)
        # propagate state to new modules added
        # TODO: there's more efficient ways to do this
        get_and_update_stateful_components(self, self.state)

    def add_link(
        self,
        src: str,
        dest: str,
        src_key: Optional[str] = None,
        dest_key: Optional[str] = None,
        condition_fn: Optional[Callable] = None,
        input_fn: Optional[Callable] = None,
    ) -> None:
        """Add a link between two modules."""
        if src not in self.module_dict:
            raise ValueError(f"Module {src} does not exist in pipeline.")
        self.dag.add_edge(
            src,
            dest,
            src_key=src_key,
            dest_key=dest_key,
            condition_fn=condition_fn,
            input_fn=input_fn,
        )

    def get_root_keys(self) -> List[str]:
        """Get root keys."""
        return self._get_root_keys()

    def get_leaf_keys(self) -> List[str]:
        """Get leaf keys."""
        return self._get_leaf_keys()

    def _get_root_keys(self) -> List[str]:
        """Get root keys."""
        return [v for v, d in self.dag.in_degree() if d == 0]

    def _get_leaf_keys(self) -> List[str]:
        """Get leaf keys."""
        # get all modules without downstream dependencies
        return [v for v, d in self.dag.out_degree() if d == 0]

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # go through every module in module dict and set callback manager
        self.callback_manager = callback_manager
        for module in self.module_dict.values():
            module.set_callback_manager(callback_manager)

    @dispatcher.span
    def run(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        batch: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the pipeline."""
        # first set callback manager
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            # try to get query payload
            try:
                query_payload = json.dumps(kwargs)
            except TypeError:
                query_payload = json.dumps(str(kwargs))
            with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_payload}
            ) as query_event:
                outputs, _ = self._run(
                    *args,
                    return_values_direct=return_values_direct,
                    show_intermediates=False,
                    batch=batch,
                    **kwargs,
                )

                return outputs

    def run_with_intermediates(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        batch: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
        """Run the pipeline."""
        if batch is not None:
            raise ValueError("Batch is not supported for run_with_intermediates.")

        # first set callback manager
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            # try to get query payload
            try:
                query_payload = json.dumps(kwargs)
            except TypeError:
                query_payload = json.dumps(str(kwargs))
            with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_payload}
            ) as query_event:
                return self._run(
                    *args,
                    return_values_direct=return_values_direct,
                    show_intermediates=True,
                    **kwargs,
                )

    def merge_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries recursively, combining values of the same key into a list."""
        merged = {}
        for key in set(d1).union(d2):
            if key in d1 and key in d2:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    merged[key] = self.merge_dicts(d1[key], d2[key])
                else:
                    new_val = [d1[key]] if not isinstance(d1[key], list) else d1[key]
                    assert isinstance(new_val, list)

                    new_val.append(d2[key])
                    merged[key] = new_val  # type: ignore[assignment]
            else:
                merged[key] = d1.get(key, d2.get(key))
        return merged

    def run_multi(
        self,
        module_input_dict: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None,
        batch: bool = False,
    ) -> Dict[str, Any]:
        """Run the pipeline for multiple roots."""
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.QUERY,
                payload={EventPayload.QUERY_STR: json.dumps(module_input_dict)},
            ) as query_event:
                if batch:
                    outputs: Dict[str, Any] = {}

                    batch_lengths = {
                        len(values)
                        for subdict in module_input_dict.values()
                        for values in subdict.values()
                    }

                    if len(batch_lengths) != 1:
                        raise ValueError("Length of batch inputs must be the same.")

                    batch_size = next(iter(batch_lengths))

                    # List individual outputs from batch multi input.
                    inputs = [
                        {
                            key: {
                                inner_key: inner_val[i]
                                for inner_key, inner_val in value.items()
                            }
                            for key, value in module_input_dict.items()
                        }
                        for i in range(batch_size)
                    ]
                    jobs = [self._arun_multi(input) for input in inputs]
                    results = asyncio_run(run_jobs(jobs, workers=len(jobs)))

                    for result in results:
                        outputs = self.merge_dicts(outputs, result[0])

                    return outputs
                else:
                    outputs, _ = self._run_multi(module_input_dict)
                    return outputs

    def run_multi_with_intermediates(
        self,
        module_input_dict: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, ComponentIntermediates]]:
        """Run the pipeline for multiple roots."""
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.QUERY,
                payload={EventPayload.QUERY_STR: json.dumps(module_input_dict)},
            ) as query_event:
                return self._run_multi(module_input_dict, show_intermediates=True)

    @dispatcher.span
    async def arun(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        batch: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the pipeline."""
        # first set callback manager
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            try:
                query_payload = json.dumps(kwargs)
            except TypeError:
                query_payload = json.dumps(str(kwargs))
            with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_payload}
            ) as query_event:
                outputs, _ = await self._arun(
                    *args,
                    return_values_direct=return_values_direct,
                    show_intermediates=False,
                    batch=batch,
                    **kwargs,
                )

                return outputs

    async def arun_with_intermediates(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        batch: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
        """Run the pipeline."""
        if batch is not None:
            raise ValueError("Batch is not supported for run_with_intermediates.")

        # first set callback manager
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            try:
                query_payload = json.dumps(kwargs)
            except TypeError:
                query_payload = json.dumps(str(kwargs))
            with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_payload}
            ) as query_event:
                return await self._arun(
                    *args,
                    return_values_direct=return_values_direct,
                    show_intermediates=True,
                    **kwargs,
                )

    async def arun_multi(
        self,
        module_input_dict: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None,
        batch: bool = False,
    ) -> Dict[str, Any]:
        """Run the pipeline for multiple roots."""
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.QUERY,
                payload={EventPayload.QUERY_STR: json.dumps(module_input_dict)},
            ) as query_event:
                if batch:
                    outputs: Dict[str, Any] = {}

                    batch_lengths = {
                        len(values)
                        for subdict in module_input_dict.values()
                        for values in subdict.values()
                    }

                    if len(batch_lengths) != 1:
                        raise ValueError("Length of batch inputs must be the same.")

                    batch_size = next(iter(batch_lengths))

                    # List individual outputs from batch multi input.
                    inputs = [
                        {
                            key: {
                                inner_key: inner_val[i]
                                for inner_key, inner_val in value.items()
                            }
                            for key, value in module_input_dict.items()
                        }
                        for i in range(batch_size)
                    ]

                    jobs = [self._arun_multi(input) for input in inputs]
                    results = await run_jobs(jobs, workers=len(jobs))

                    for result in results:
                        outputs = self.merge_dicts(outputs, result[0])

                    return outputs
                else:
                    outputs, _ = await self._arun_multi(module_input_dict)
                    return outputs

    async def arun_multi_with_intermediates(
        self,
        module_input_dict: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline for multiple roots."""
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.QUERY,
                payload={EventPayload.QUERY_STR: json.dumps(module_input_dict)},
            ) as query_event:
                outputs, _ = await self._arun_multi(
                    module_input_dict, show_intermediates=True
                )
                return outputs

    def _get_root_key_and_kwargs(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get root key and kwargs.

        This is for `_run`.

        """
        ## run pipeline
        ## assume there is only one root - for multiple roots, need to specify `run_multi`
        root_keys = self._get_root_keys()
        if len(root_keys) != 1:
            raise ValueError("Only one root is supported.")
        root_key = root_keys[0]

        root_module = self.module_dict[root_key]
        if len(args) > 0:
            # if args is specified, validate. only one arg is allowed, and there can only be one free
            # input key in the module
            if len(args) > 1:
                raise ValueError("Only one arg is allowed.")
            if len(kwargs) > 0:
                raise ValueError("No kwargs allowed if args is specified.")
            if len(root_module.free_req_input_keys) != 1:
                raise ValueError("Only one free input key is allowed.")
            # set kwargs
            kwargs[next(iter(root_module.free_req_input_keys))] = args[0]

        # if one kwarg and module only needs one kwarg, align them
        if len(root_module.free_req_input_keys) == 1 and len(kwargs) == 1:
            module_input_key = next(iter(root_module.free_req_input_keys))
            kwarg = next(iter(kwargs.values()))
            kwargs = {module_input_key: kwarg}

        return root_key, kwargs

    def get_input_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Get input dict."""
        root_key, kwargs = self._get_root_key_and_kwargs(*args, **kwargs)
        return {root_key: kwargs}

    def _get_single_result_output(
        self,
        result_outputs: Dict[str, Any],
        return_values_direct: bool,
    ) -> Any:
        """
        Get result output from a single module.

        If output dict is a single key, return the value directly
        if return_values_direct is True.

        """
        if len(result_outputs) != 1:
            raise ValueError("Only one output is supported.")

        result_output = next(iter(result_outputs.values()))
        # return_values_direct: if True, return the value directly
        # without the key
        # if it's a dict with one key, return the value
        if (
            isinstance(result_output, dict)
            and len(result_output) == 1
            and return_values_direct
        ):
            return next(iter(result_output.values()))
        else:
            return result_output

    @dispatcher.span
    def _run(
        self,
        *args: Any,
        return_values_direct: bool = True,
        show_intermediates: bool = False,
        batch: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
        """
        Run the pipeline.

        Assume that there is a single root module and a single output module.

        For multi-input and multi-outputs, please see `run_multi`.

        """
        root_key, kwargs = self._get_root_key_and_kwargs(*args, **kwargs)

        if batch:
            result_outputs = []
            intermediates = []

            if len({len(value) for value in kwargs.values()}) != 1:
                raise ValueError("Length of batch inputs must be the same.")

            # List of individual inputs from batch input
            kwargs_list = [
                dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())
            ]

            jobs = [
                self._arun_multi(
                    {root_key: kwarg}, show_intermediates=show_intermediates
                )
                for kwarg in kwargs_list
            ]

            results = asyncio_run(run_jobs(jobs, workers=len(jobs)))

            for result in results:
                result_outputs.append(
                    self._get_single_result_output(result[0], return_values_direct)
                )
                intermediates.append(result[1])

            return result_outputs, intermediates  # type: ignore[return-value]
        else:
            result_output_dicts, intermediate_dicts = self._run_multi(
                {root_key: kwargs}, show_intermediates=show_intermediates
            )

            return (
                self._get_single_result_output(
                    result_output_dicts, return_values_direct
                ),
                intermediate_dicts,
            )

    @dispatcher.span
    async def _arun(
        self,
        *args: Any,
        return_values_direct: bool = True,
        show_intermediates: bool = False,
        batch: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
        """
        Run the pipeline.

        Assume that there is a single root module and a single output module.

        For multi-input and multi-outputs, please see `run_multi`.

        """
        root_key, kwargs = self._get_root_key_and_kwargs(*args, **kwargs)

        if batch:
            result_outputs = []
            intermediates = []

            if len({len(value) for value in kwargs.values()}) != 1:
                raise ValueError("Length of batch inputs must be the same.")

            # List of individual inputs from batch input
            kwargs_list = [
                dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())
            ]

            jobs = [
                self._arun_multi(
                    {root_key: kwarg}, show_intermediates=show_intermediates
                )
                for kwarg in kwargs_list
            ]

            results = await run_jobs(jobs, workers=len(jobs))

            for result in results:
                result_outputs.append(
                    self._get_single_result_output(result[0], return_values_direct)
                )
                intermediates.append(result[1])

            return result_outputs, intermediates  # type: ignore[return-value]
        else:
            result_output_dicts, intermediate_dicts = await self._arun_multi(
                {root_key: kwargs}, show_intermediates=show_intermediates
            )

            return (
                self._get_single_result_output(
                    result_output_dicts, return_values_direct
                ),
                intermediate_dicts,
            )

    def _validate_inputs(self, module_input_dict: Dict[str, Any]) -> None:
        root_keys = self._get_root_keys()
        # if root keys don't match up with kwargs keys, raise error
        if set(root_keys) != set(module_input_dict.keys()):
            raise ValueError(
                "Expected root keys do not match up with input keys.\n"
                f"Expected root keys: {root_keys}\n"
                f"Input keys: {module_input_dict.keys()}\n"
            )

    def process_component_output(
        self,
        output_dict: Dict[str, Any],
        module_key: str,
        run_state: RunState,
    ) -> None:
        """Process component output."""
        if module_key in self._get_leaf_keys():
            run_state.result_outputs[module_key] = output_dict
        else:
            edge_list = list(self.dag.edges(module_key, data=True))

            for _, dest, attr in edge_list:
                if dest in run_state.executed_modules:
                    continue  # Skip already executed modules

                output = get_output(attr.get("src_key"), output_dict)

                if attr["input_fn"] is not None:
                    dest_output = attr["input_fn"](output)
                else:
                    dest_output = output

                if attr["condition_fn"] is None or attr["condition_fn"](output):
                    add_output_to_module_inputs(
                        attr.get("dest_key"),
                        dest_output,
                        self.module_dict[dest],
                        run_state.all_module_inputs[dest],
                    )

        run_state.executed_modules.add(module_key)

    def get_next_module_keys(self, run_state: RunState) -> List[str]:
        """Determine the next module keys to run based on the current state."""
        next_module_keys = []

        for module_key, module_input in run_state.all_module_inputs.items():
            if module_key in run_state.executed_modules:
                continue  # Module already executed

            if all(
                key in module_input
                for key in self.module_dict[module_key].free_req_input_keys
            ):
                next_module_keys.append(module_key)

        return next_module_keys

    def get_run_state(
        self, module_input_dict: Optional[Dict[str, Any]] = None, **pipeline_inputs: Any
    ) -> RunState:
        """Get run state."""
        if module_input_dict is not None:
            return RunState(self.module_dict, module_input_dict)
        else:
            root_key, kwargs = self._get_root_key_and_kwargs(**pipeline_inputs)
            return RunState(self.module_dict, {root_key: kwargs})

    @dispatcher.span
    def _run_multi(
        self, module_input_dict: Dict[str, Any], show_intermediates: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, ComponentIntermediates]]:
        """Run the pipeline for multiple roots."""
        self._validate_inputs(module_input_dict)

        run_state = self.get_run_state(module_input_dict)

        # Add root inputs to all_module_inputs
        next_module_keys = self.get_next_module_keys(run_state)

        while True:
            for module_key in next_module_keys:
                module = run_state.module_dict[module_key]
                module_input = run_state.all_module_inputs[module_key]

                if self.verbose:
                    print_debug_input(module_key, module_input)
                output_dict = module.run_component(**module_input)

                if (
                    show_intermediates
                    and module_key not in run_state.intermediate_outputs
                ):
                    run_state.intermediate_outputs[module_key] = ComponentIntermediates(
                        inputs=module_input, outputs=output_dict
                    )

                self.process_component_output(
                    output_dict,
                    module_key,
                    run_state,
                )

            next_module_keys = self.get_next_module_keys(
                run_state,
            )
            if not next_module_keys:
                run_state.result_outputs[module_key] = output_dict
                break

        return run_state.result_outputs, run_state.intermediate_outputs

    @dispatcher.span
    async def _arun_multi(
        self, module_input_dict: Dict[str, Any], show_intermediates: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, ComponentIntermediates]]:
        """
        Run the pipeline for multiple roots.

        kwargs is in the form of module_dict -> input_dict
        input_dict is in the form of input_key -> input

        """
        self._validate_inputs(module_input_dict)

        run_state = self.get_run_state(module_input_dict)

        # Add root inputs to all_module_inputs
        next_module_keys = self.get_next_module_keys(run_state)

        while True:
            jobs = []
            for module_key in next_module_keys:
                module = run_state.module_dict[module_key]
                module_input = run_state.all_module_inputs[module_key]

                if self.verbose:
                    print_debug_input(module_key, module_input)

                jobs.append(module.arun_component(**module_input))

            output_dicts = await run_jobs(jobs, show_progress=self.show_progress)
            for module_key, output_dict in zip(next_module_keys, output_dicts):
                if (
                    show_intermediates
                    and module_key not in run_state.intermediate_outputs
                ):
                    run_state.intermediate_outputs[module_key] = ComponentIntermediates(
                        inputs=module_input, outputs=output_dict
                    )

                self.process_component_output(
                    output_dict,
                    module_key,
                    run_state,
                )

            next_module_keys = self.get_next_module_keys(
                run_state,
            )
            if not next_module_keys:
                run_state.result_outputs[module_key] = output_dicts[-1]
                break

        return run_state.result_outputs, run_state.intermediate_outputs

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        raise NotImplementedError

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs."""
        return input

    def _validate_component_outputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # NOTE: we override this to do nothing
        return output

    def _run_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        return self.run(return_values_direct=False, **kwargs)

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        return await self.arun(return_values_direct=False, **kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # get input key of first module
        root_keys = self._get_root_keys()
        if len(root_keys) != 1:
            raise ValueError("Only one root is supported.")
        root_module = self.module_dict[root_keys[0]]
        return root_module.input_keys

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        # get output key of last module
        leaf_keys = self._get_leaf_keys()
        if len(leaf_keys) != 1:
            raise ValueError("Only one leaf is supported.")
        leaf_module = self.module_dict[leaf_keys[0]]
        return leaf_module.output_keys

    @property
    def sub_query_components(self) -> List[QueryComponent]:
        """Sub query components."""
        return list(self.module_dict.values())

    @property
    def clean_dag(self) -> networkx.DiGraph:
        """Clean dag."""
        return clean_graph_attributes_copy(self.dag)
