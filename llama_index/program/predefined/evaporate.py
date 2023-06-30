import logging
from typing import Any, Dict, List, Type

from llama_index.experimental.evaporate.base import EvaporateExtractor
from llama_index.program.base_program import BasePydanticProgram
from llama_index.program.predefined.df import DataFrameRow
from llama_index.schema import BaseNode

logger = logging.getLogger(__name__)


class EvaporateProgram(BasePydanticProgram[DataFrameRow]):
    """Evaporate program. You should provide the fields you want to extract.
    Then when you call the program you should pass in a list of training_data nodes
    and a list of infer_data nodes. The program will call the EvaporateExtractor
    to synthesize a python function from the training data and then apply the function
    to the infer_data.
    """

    def __init__(
        self,
        fields_to_extract: List[str],
        training_key: str = "training_data",
        infer_key: str = "infer_data",
        **program_kwargs: Any,
    ) -> None:
        """Init params."""
        self._extractor = EvaporateExtractor()
        self._field_fns: Dict[str, str] = {}
        self._fields = fields_to_extract
        self._training_key = training_key
        self._infer_key = infer_key

    def _fit(self, nodes: List[BaseNode], field: str) -> str:
        """Given the input Nodes and fields, synthesize the python code."""
        fn = self._extractor.extract_fn_from_nodes(nodes, field)
        logger.debug(f"Extracted function: {fn}")
        return fn

    def _inference(self, nodes: List[BaseNode], fn_str: str, field_name: str) -> str:
        """Given the input, call the python code and return the result."""
        results = self._extractor.run_fn_on_nodes(nodes, fn_str, field_name)
        logger.debug(f"Results: {results}")
        return str(results)

    @classmethod
    def from_defaults(
        cls,
        fields_to_extract: List[str],
        **kwargs: Any,
    ) -> "EvaporateProgram":
        """Evaporate program."""

        return cls(
            fields_to_extract,
            **kwargs,
        )

    @property
    def output_cls(self) -> Type[DataFrameRow]:
        """Output class."""
        return DataFrameRow

    def __call__(self, *args: Any, **kwds: Any) -> DataFrameRow:
        """Call evaporate on training and inference data. Inputs should be two
        lists of nodes, using the keys `training_data` and `infer_data`."""
        for field in self._fields:
            self._field_fns[field] = self._fit(kwds[self._training_key], field)
        infer_vals = []
        for field in self._fields:
            infer_vals.append(
                self._inference(kwds[self._infer_key], self._field_fns[field], field)
            )
        result = DataFrameRow(row_values=infer_vals)
        return result
