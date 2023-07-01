import logging
from typing import Any, Dict, List, Type, Optional

from llama_index.experimental.evaporate.base import EvaporateExtractor
from llama_index.program.base_program import BasePydanticProgram
from llama_index.program.predefined.df import DataFrameRowsOnly, DataFrameRow
from llama_index.schema import BaseNode
from llama_index.indices.service_context import ServiceContext
from llama_index.program.predefined.evaporate.prompts import (
    FN_GENERATION_PROMPT,
    SCHEMA_ID_PROMPT,
    FnGeneratePrompt,
    SchemaIDPrompt,
    DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
)


logger = logging.getLogger(__name__)


class EvaporateProgram(BasePydanticProgram[DataFrameRowsOnly]):
    """Evaporate program.

    You should provide the fields you want to extract.
    Then when you call the program you should pass in a list of training_data nodes
    and a list of infer_data nodes. The program will call the EvaporateExtractor
    to synthesize a python function from the training data and then apply the function
    to the infer_data.
    """

    def __init__(
        self,
        extractor: EvaporateExtractor,
        fields_to_extract: Optional[List[str]] = None,
        nodes_to_fit: Optional[List[BaseNode]] = None,
        infer_key: str = "infer_data",
        **program_kwargs: Any,
    ) -> None:
        """Init params."""
        self._extractor = extractor
        self._fields = fields_to_extract or []
        # NOTE: this will change with each call to `fit`
        self._field_fns: Dict[str, str] = {}
        self._infer_key = infer_key

        # if nodes_to_fit is not None, then fit extractor
        if nodes_to_fit is not None:
            self._field_fns = self.fit_fields(nodes_to_fit)

    @classmethod
    def from_defaults(
        cls,
        fields_to_extract: Optional[List[str]] = None,
        service_context: Optional[ServiceContext] = None,
        schema_id_prompt: Optional[SchemaIDPrompt] = None,
        fn_generate_prompt: Optional[FnGeneratePrompt] = None,
        field_extract_query_tmpl: str = DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
        nodes_to_fit: Optional[List[BaseNode]] = None,
        inference_key: str = "input_str",
        **program_kwargs: Any,
    ) -> "EvaporateProgram":
        """Evaporate program."""
        extractor = EvaporateExtractor(
            service_context=service_context,
            schema_id_prompt=schema_id_prompt,
            fn_generate_prompt=fn_generate_prompt,
        )
        return cls(
            extractor,
            fields_to_extract=fields_to_extract,
            nodes_to_fit=nodes_to_fit,
            inference_key=inference_key,
            **program_kwargs,
        )

    @property
    def extractor(self) -> EvaporateExtractor:
        """Extractor."""
        return self._extractor

    def get_function_str(self, field: str) -> str:
        """Get function string."""
        return self._field_fns[field]

    def set_fields_to_extract(self, fields: List[str]) -> None:
        """Set fields to extract."""
        self._fields = fields

    def fit_fields(
        self,
        nodes: List[BaseNode],
        inplace: bool = True,
    ) -> Dict[str, str]:
        """Fit on a set of fields."""
        if len(self._fields) == 0:
            raise ValueError("Must provide at least one field to extract.")

        field_fns = {}
        for field in self._fields:
            field_fns[field] = self.fit(nodes, field, inplace=inplace)
        return field_fns

    def fit(self, nodes: List[BaseNode], field: str, inplace: bool = True) -> str:
        """Given the input Nodes and fields, synthesize the python code."""
        fn = self._extractor.extract_fn_from_nodes(nodes, field)
        logger.debug(f"Extracted function: {fn}")
        if inplace:
            self._field_fns[field] = fn
        return fn

    def _inference(self, nodes: List[BaseNode], fn_str: str, field_name: str) -> str:
        """Given the input, call the python code and return the result."""
        results = self._extractor.run_fn_on_nodes(nodes, fn_str, field_name)
        logger.debug(f"Results: {results}")
        return str(results)

    @property
    def output_cls(self) -> Type[DataFrameRowsOnly]:
        """Output class."""
        return DataFrameRowsOnly

    def __call__(self, *args: Any, **kwds: Any) -> DataFrameRowsOnly:
        """Call evaporate on inference data."""

        infer_vals = []
        for field in self._fields:
            infer_vals.append(
                self._inference(kwds[self._infer_key], self._field_fns[field], field)
            )
        data_frame_row = DataFrameRow(row_values=infer_vals)
        result = DataFrameRowsOnly(rows=[data_frame_row])
        return result
