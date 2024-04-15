"""Query Pipeline."""

import json
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
)

import networkx

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import Field
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


CHAIN_COMPONENT_TYPE = Union[QUERY_COMPONENT_TYPE, str]


class QueryPipeline(QueryComponent):
    """A query pipeline that can allow arbitrary chaining of different modules.

    A pipeline itself is a query component, and can be used as a module in another pipeline.

    """

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

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        chain: Optional[Sequence[CHAIN_COMPONENT_TYPE]] = None,
        modules: Optional[Dict[str, QUERY_COMPONENT_TYPE]] = None,
        links: Optional[List[Link]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            callback_manager=callback_manager or CallbackManager([]),
            **kwargs,
        )

        self._init_graph(chain=chain, modules=modules, links=links)

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
                    self.add_link(**link.dict())

    def add_chain(self, chain: Sequence[CHAIN_COMPONENT_TYPE]) -> None:
        """Add a chain of modules to the pipeline.

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

    def add_links(
        self,
        links: List[Link],
    ) -> None:
        """Add links to the pipeline."""
        for link in links:
            if isinstance(link, Link):
                self.add_link(**link.dict())
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

    def run(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
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
                    **kwargs,
                )
                return outputs

    def run_with_intermediates(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
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
                return self._run(
                    *args,
                    return_values_direct=return_values_direct,
                    show_intermediates=True,
                    **kwargs,
                )

    def run_multi(
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

    async def arun(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
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
                    **kwargs,
                )
                return outputs

    async def arun_with_intermediates(
        self,
        *args: Any,
        return_values_direct: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
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
    ) -> Dict[str, Any]:
        """Run the pipeline for multiple roots."""
        callback_manager = callback_manager or self.callback_manager
        self.set_callback_manager(callback_manager)
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.QUERY,
                payload={EventPayload.QUERY_STR: json.dumps(module_input_dict)},
            ) as query_event:
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
                return await self._arun_multi(
                    module_input_dict, show_intermediates=True
                )

    def _get_root_key_and_kwargs(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """Get root key and kwargs.

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
        return root_key, kwargs

    def _get_single_result_output(
        self,
        result_outputs: Dict[str, Any],
        return_values_direct: bool,
    ) -> Any:
        """Get result output from a single module.

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

    def _run(
        self,
        *args: Any,
        return_values_direct: bool = True,
        show_intermediates: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
        """Run the pipeline.

        Assume that there is a single root module and a single output module.

        For multi-input and multi-outputs, please see `run_multi`.

        """
        root_key, kwargs = self._get_root_key_and_kwargs(*args, **kwargs)

        result_outputs, intermediates = self._run_multi(
            {root_key: kwargs}, show_intermediates=show_intermediates
        )

        return (
            self._get_single_result_output(result_outputs, return_values_direct),
            intermediates,
        )

    async def _arun(
        self,
        *args: Any,
        return_values_direct: bool = True,
        show_intermediates: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, ComponentIntermediates]]:
        """Run the pipeline.

        Assume that there is a single root module and a single output module.

        For multi-input and multi-outputs, please see `run_multi`.

        """
        root_key, kwargs = self._get_root_key_and_kwargs(*args, **kwargs)

        result_outputs, intermediates = await self._arun_multi(
            {root_key: kwargs}, show_intermediates=show_intermediates
        )

        return (
            self._get_single_result_output(result_outputs, return_values_direct),
            intermediates,
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

    def _process_component_output(
        self,
        queue: List[str],
        output_dict: Dict[str, Any],
        module_key: str,
        all_module_inputs: Dict[str, Dict[str, Any]],
        result_outputs: Dict[str, Any],
    ) -> List[str]:
        """Process component output."""
        new_queue = queue.copy()

        nodes_to_keep = set()
        nodes_to_remove = set()

        # if there's no more edges, clear queue
        if module_key in self._get_leaf_keys():
            new_queue = []
        else:
            edge_list = list(self.dag.edges(module_key, data=True))

            # everything not in conditional_edge_list is regular
            for _, dest, attr in edge_list:
                output = get_output(attr.get("src_key"), output_dict)

                # if input_fn is not None, use it to modify the input
                if attr["input_fn"] is not None:
                    dest_output = attr["input_fn"](output)
                else:
                    dest_output = output

                add_edge = True
                if attr["condition_fn"] is not None:
                    conditional_val = attr["condition_fn"](output)
                    if not conditional_val:
                        add_edge = False

                if add_edge:
                    add_output_to_module_inputs(
                        attr.get("dest_key"),
                        dest_output,
                        self.module_dict[dest],
                        all_module_inputs[dest],
                    )
                    nodes_to_keep.add(dest)
                else:
                    nodes_to_remove.add(dest)

        # remove nodes from the queue, as well as any nodes that depend on dest
        # be sure to not remove any remaining dependencies of the current path
        available_paths = []
        for node in nodes_to_keep:
            for leaf_node in self._get_leaf_keys():
                if leaf_node == node:
                    available_paths.append([node])
                else:
                    available_paths.extend(
                        list(
                            networkx.all_simple_paths(
                                self.dag, source=node, target=leaf_node
                            )
                        )
                    )

        # this is a list of all nodes between the current node(s) and the leaf nodes
        nodes_to_never_remove = set(x for path in available_paths for x in path)  # noqa

        removal_paths = []
        for node in nodes_to_remove:
            for leaf_node in self._get_leaf_keys():
                if leaf_node == node:
                    removal_paths.append([node])
                else:
                    removal_paths.extend(
                        list(
                            networkx.all_simple_paths(
                                self.dag, source=node, target=leaf_node
                            )
                        )
                    )

        # this is a list of all nodes between the current node(s) to remove and the leaf nodes
        nodes_to_probably_remove = set(  # noqa
            x for path in removal_paths for x in path
        )

        # remove nodes that are not in the current path
        for node in nodes_to_probably_remove:
            if node not in nodes_to_never_remove:
                new_queue.remove(node)

        # did we remove all remaining edges? then we have our result
        if len(new_queue) == 0:
            result_outputs[module_key] = output_dict

        return new_queue

    def _run_multi(
        self, module_input_dict: Dict[str, Any], show_intermediates=False
    ) -> Tuple[Dict[str, Any], Dict[str, ComponentIntermediates]]:
        """Run the pipeline for multiple roots.

        kwargs is in the form of module_dict -> input_dict
        input_dict is in the form of input_key -> input

        """
        self._validate_inputs(module_input_dict)
        queue = list(networkx.topological_sort(self.dag))

        # module_deps_inputs is a dict to collect inputs for a module
        # mapping of module_key -> dict of input_key -> input
        # initialize with blank dict for every module key
        # the input dict of each module key will be populated as the upstream modules are run
        all_module_inputs: Dict[str, Dict[str, Any]] = {
            module_key: {} for module_key in self.module_dict
        }
        result_outputs: Dict[str, Any] = {}
        intermediate_outputs: Dict[str, ComponentIntermediates] = {}

        # add root inputs to all_module_inputs
        for module_key, module_input in module_input_dict.items():
            all_module_inputs[module_key] = module_input

        while len(queue) > 0:
            module_key = queue.pop(0)
            module = self.module_dict[module_key]
            module_input = all_module_inputs[module_key]

            if self.verbose:
                print_debug_input(module_key, module_input)
            output_dict = module.run_component(**module_input)

            if show_intermediates and module_key not in intermediate_outputs:
                intermediate_outputs[module_key] = ComponentIntermediates(
                    inputs=module_input, outputs=output_dict
                )

            # get new nodes and is_leaf
            queue = self._process_component_output(
                queue, output_dict, module_key, all_module_inputs, result_outputs
            )

        return result_outputs, intermediate_outputs

    async def _arun_multi(
        self, module_input_dict: Dict[str, Any], show_intermediates: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, ComponentIntermediates]]:
        """Run the pipeline for multiple roots.

        kwargs is in the form of module_dict -> input_dict
        input_dict is in the form of input_key -> input

        """
        self._validate_inputs(module_input_dict)
        queue = list(networkx.topological_sort(self.dag))

        # module_deps_inputs is a dict to collect inputs for a module
        # mapping of module_key -> dict of input_key -> input
        # initialize with blank dict for every module key
        # the input dict of each module key will be populated as the upstream modules are run
        all_module_inputs: Dict[str, Dict[str, Any]] = {
            module_key: {} for module_key in self.module_dict
        }
        result_outputs: Dict[str, Any] = {}
        intermediate_outputs: Dict[str, ComponentIntermediates] = {}

        # add root inputs to all_module_inputs
        for module_key, module_input in module_input_dict.items():
            all_module_inputs[module_key] = module_input

        while len(queue) > 0:
            popped_indices = set()
            popped_nodes = []
            # get subset of nodes who don't have ancestors also in the queue
            # these are tasks that are parallelizable
            for i, module_key in enumerate(queue):
                module_ancestors = networkx.ancestors(self.dag, module_key)
                if len(set(module_ancestors).intersection(queue)) == 0:
                    popped_indices.add(i)
                    popped_nodes.append(module_key)

            # update queue
            queue = [
                module_key
                for i, module_key in enumerate(queue)
                if i not in popped_indices
            ]

            if self.verbose:
                print_debug_input_multi(
                    popped_nodes,
                    [all_module_inputs[module_key] for module_key in popped_nodes],
                )

            if show_intermediates and module_key not in intermediate_outputs:
                intermediate_outputs[module_key] = ComponentIntermediates(
                    inputs=module_input, outputs=output_dict
                )

            # create tasks from popped nodes
            tasks = []
            for module_key in popped_nodes:
                module = self.module_dict[module_key]
                module_input = all_module_inputs[module_key]
                tasks.append(module.arun_component(**module_input))

            # run tasks
            output_dicts = await run_jobs(
                tasks, show_progress=self.show_progress, workers=self.num_workers
            )

            for output_dict, module_key in zip(output_dicts, popped_nodes):
                # get new nodes and is_leaf
                queue = self._process_component_output(
                    queue, output_dict, module_key, all_module_inputs, result_outputs
                )

        return result_outputs, intermediate_outputs

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
