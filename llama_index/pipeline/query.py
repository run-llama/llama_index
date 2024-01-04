"""Query Pipeline."""

from typing import Any, List, Optional, Sequence, Dict, Set
from llama_index.bridge.pydantic import Field, PrivateAttr, BaseModel, validator
from llama_index.callbacks import CallbackManager
from llama_index.schema import BaseComponent
from llama_index.pipeline.schema import QueryComponent
import uuid


class InputTup(BaseModel):
    """Input Tuple."""
    module_key: str
    module: QueryComponent
    input: Dict[str, Any]
    


class Link(BaseModel):
    """Link between two modules."""
    src: str
    dest: str
    src_key: Optional[str] = None
    dest_key: Optional[str] = None


# class ModuleDepsInputs(BaseModel):
#     """Module Inputs from Upstream Dependencies."""
#     module_key: str
#     input_dict: Dict[str, Any]

def add_output_to_module_inputs(
    link: Link, 
    output_dict: Dict[str, Any], 
    module_input_keys: Set[str], 
    module_inputs: Dict[str, Any]
) -> None:
    """Add input to module deps inputs."""
    
    # get relevant output from link
    if link.src_key is None:
        # ensure that output_dict only has one key
        if len(output_dict) != 1:
            raise ValueError("Output dict must have exactly one key.")
        output = list(output_dict.values())[0]
    else:
        output = output_dict[link.src_key]

    # now attach output to relevant input key for module
    if link.dest_key is None:
        # ensure that module_input_keys only has one key
        if len(module_input_keys) != 1:
            raise ValueError("Module input keys must have exactly one key.")
        module_inputs[module_input_keys[0]] = output
    else:
        module_inputs[link.dest_key] = output

class QueryPipeline(BaseModel):
    """A query pipeline that can allow arbitrary chaining of different modules."""

    callback_manager = Field(
        default_factory=CallbackManager, exclude=True
    )

    module_dict: Dict[str, Any] = Field(
        default_factory=dict, description="The modules in the pipeline."
    )
    edge_dict: Dict[str, List[Link]] = Field(
        default_factory=dict, description="The edges in the pipeline."
    )

    # root_keys: List[str] = Field(
    #     default_factory=list, description="The keys of the root modules."
    # )

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        chain: Optional[Sequence[QueryComponent]] = None,
        **kwargs: Any,
    ):
        callback_manager = callback_manager or CallbackManager([])
        
        super().__init__(
            callback_manager=callback_manager,
            **kwargs,
        )

        if chain is not None:
            # generate implicit link between each item, add 
            self._add_chain(chain)

    def add_chain(self, chain: Sequence[QueryComponent]) -> None:
        """Add a chain of modules to the pipeline.

        This is a special form of pipeline that is purely sequential/linear.
        This allows a more concise way of specifying a pipeline.
        
        """
        # first add all modules
        module_keys = []
        for module in chain:
            module_key = str(uuid.uuid4())
            self.add(module_key, module)

        # then add all links
        for i in range(len(chain) - 1):
            self.add_link(
                src=module_keys[i], 
                dest=module_keys[i + 1]
            )

    def add_modules(self, module_dict: Dict[str, QueryComponent]) -> None:
        """Add modules to the pipeline."""
        for module_key, module in module_dict.items():
            self.add(module_key, module)

    def add(self, module_key: str, module: QueryComponent) -> None:
        """Add a module to the pipeline."""
        # if already exists, raise error
        if module_key in self.module_dict:
            raise ValueError(f"Module {module_key} already exists in pipeline.")
        self.module_dict[module_key] = module

    def add_link(
        self, 
        src: str, 
        dest: str, 
        src_key: Optional[str] = None, 
        dest_key: Optional[str] = None
    ) -> None:
        """Add a link between two modules."""
        self.edge_dict[src].append(
            Link(src=src, dest=dest, src_key=src_key, dest_key=dest_key)
        )

    def _get_root_keys(self) -> List[str]:
        """Get root keys."""
        # first add all to root keys, then remove
        root_keys: Set[str] = set(self.module_dict.keys())
        # get all modules without upstream dependencies
        for module_key in self.module_dict.keys():
            for link in self.edge_dict[module_key]:
                if link.dest in root_keys:
                    root_keys.remove(link.dest)
        return root_keys

    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline.

        Assume that there is a single root module and a single output module.

        For multi-input and multi-outputs, please see `run_multi`.
        
        """
        # currently assume kwargs, add handling for args later
        ## run pipeline
        ## assume there is only one root - for multiple roots, need to specify `run_multi`
        root_keys = self._get_root_keys()
        if len(root_keys) != 1:
            raise ValueError("Only one root is supported.")
        root_module = self.module_dict[root_keys[0]]

        # for each module in queue, run it, and then check if its edges have all their dependencies satisfied
        # if so, add to queue
        queue: List[InputTup] = [InputTup(key=self.root_keys[0], module=root_module, input=kwargs)]

        # module_deps_inputs is a dict to collect inputs for a module 
        # mapping of module_key -> dict of input_key -> input
        # initialize with blank dict for every module key
        # the input dict of each module key will be populated as the upstream modules are run
        all_module_inputs: Dict[str, Dict[str, Any]] = {
            {} for module_key in self.module_dict.keys()
        }
        result_outputs: List[Any] = []
        while len(queue) > 0:
            input_tup = queue.pop(0)
            module_key, module, input = input_tup.module_key, input_tup.module, input_tup.input

            output_dict = module.run_component(**input)
            
            # if there's no more edges, add result to output
            if module_key not in self.edge_dict:
                result_outputs.append(output_dict)
            else:
                for link in self.edge_dict[module_key]:
                    edge_module = self.module_dict[link.dest]

                    # add input to module_deps_inputs
                    add_output_to_module_inputs(
                        link, output_dict, edge_module.input_keys, all_module_inputs[link.dest]
                    )
                    if len(all_module_inputs[link.dest]) == len(edge_module.input_keys):
                        queue.append(
                            InputTup(
                                module_key=link.dest,
                                module=edge_module,
                                input=all_module_inputs[link.dest],
                            )
                        )

        if len(result_outputs) != 1:
            raise ValueError("Only one output is supported.")

        result_output = result_outputs[0]
        # if it's a dict with one key, return the value
        if isinstance(result_output, dict) and len(result_output) == 1:
            return list(result_output.values())[0]
        else:
            return result_output
        

    def run_multi(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run the pipeline for multiple roots."""
        raise NotImplementedError("Not implemented yet.")
        

    
    
    
    