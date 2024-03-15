import logging
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type

import pandas as pd

from llama_index.legacy.program.predefined.df import (
    DataFrameRow,
    DataFrameRowsOnly,
    DataFrameValuesPerColumn,
)
from llama_index.legacy.program.predefined.evaporate.extractor import EvaporateExtractor
from llama_index.legacy.program.predefined.evaporate.prompts import (
    DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
    FN_GENERATION_LIST_PROMPT,
    FnGeneratePrompt,
    SchemaIDPrompt,
)
from llama_index.legacy.schema import BaseNode, TextNode
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.types import BasePydanticProgram, Model
from llama_index.legacy.utils import print_text

logger = logging.getLogger(__name__)


class BaseEvaporateProgram(BasePydanticProgram, Generic[Model]):
    """BaseEvaporate program.

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
        fields_context: Optional[Dict[str, Any]] = None,
        nodes_to_fit: Optional[List[BaseNode]] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        self._extractor = extractor
        self._fields = fields_to_extract or []
        self._fields_context = fields_context or {}
        # NOTE: this will change with each call to `fit`
        self._field_fns: Dict[str, str] = {}
        self._verbose = verbose

        # if nodes_to_fit is not None, then fit extractor
        if nodes_to_fit is not None:
            self._field_fns = self.fit_fields(nodes_to_fit)

    @classmethod
    def from_defaults(
        cls,
        fields_to_extract: Optional[List[str]] = None,
        fields_context: Optional[Dict[str, Any]] = None,
        service_context: Optional[ServiceContext] = None,
        schema_id_prompt: Optional[SchemaIDPrompt] = None,
        fn_generate_prompt: Optional[FnGeneratePrompt] = None,
        field_extract_query_tmpl: str = DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
        nodes_to_fit: Optional[List[BaseNode]] = None,
        verbose: bool = False,
    ) -> "BaseEvaporateProgram":
        """Evaporate program."""
        extractor = EvaporateExtractor(
            service_context=service_context,
            schema_id_prompt=schema_id_prompt,
            fn_generate_prompt=fn_generate_prompt,
            field_extract_query_tmpl=field_extract_query_tmpl,
        )
        return cls(
            extractor,
            fields_to_extract=fields_to_extract,
            fields_context=fields_context,
            nodes_to_fit=nodes_to_fit,
            verbose=verbose,
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
        """Fit on all fields."""
        if len(self._fields) == 0:
            raise ValueError("Must provide at least one field to extract.")

        field_fns = {}
        for field in self._fields:
            field_context = self._fields_context.get(field, None)
            field_fns[field] = self.fit(
                nodes, field, field_context=field_context, inplace=inplace
            )
        return field_fns

    @abstractmethod
    def fit(
        self,
        nodes: List[BaseNode],
        field: str,
        field_context: Optional[Any] = None,
        expected_output: Optional[Any] = None,
        inplace: bool = True,
    ) -> str:
        """Given the input Nodes and fields, synthesize the python code."""


class DFEvaporateProgram(BaseEvaporateProgram[DataFrameRowsOnly]):
    """Evaporate DF program.

    Given a set of fields, extracts a dataframe from a set of nodes.
    Each node corresponds to a row in the dataframe - each value in the row
    corresponds to a field value.

    """

    def fit(
        self,
        nodes: List[BaseNode],
        field: str,
        field_context: Optional[Any] = None,
        expected_output: Optional[Any] = None,
        inplace: bool = True,
    ) -> str:
        """Given the input Nodes and fields, synthesize the python code."""
        fn = self._extractor.extract_fn_from_nodes(nodes, field)
        logger.debug(f"Extracted function: {fn}")
        if inplace:
            self._field_fns[field] = fn
        return fn

    def _inference(
        self, nodes: List[BaseNode], fn_str: str, field_name: str
    ) -> List[Any]:
        """Given the input, call the python code and return the result."""
        results = self._extractor.run_fn_on_nodes(nodes, fn_str, field_name)
        logger.debug(f"Results: {results}")
        return results

    @property
    def output_cls(self) -> Type[DataFrameRowsOnly]:
        """Output class."""
        return DataFrameRowsOnly

    def __call__(self, *args: Any, **kwds: Any) -> DataFrameRowsOnly:
        """Call evaporate on inference data."""
        # TODO: either specify `nodes` or `texts` in kwds
        if "nodes" in kwds:
            nodes = kwds["nodes"]
        elif "texts" in kwds:
            nodes = [TextNode(text=t) for t in kwds["texts"]]
        else:
            raise ValueError("Must provide either `nodes` or `texts`.")

        col_dict = {}
        for field in self._fields:
            col_dict[field] = self._inference(nodes, self._field_fns[field], field)

        df = pd.DataFrame(col_dict, columns=self._fields)

        # convert pd.DataFrame to DataFrameRowsOnly
        df_row_objs = []
        for row_arr in df.values:
            df_row_objs.append(DataFrameRow(row_values=list(row_arr)))
        return DataFrameRowsOnly(rows=df_row_objs)


class MultiValueEvaporateProgram(BaseEvaporateProgram[DataFrameValuesPerColumn]):
    """Multi-Value Evaporate program.

    Given a set of fields, and texts extracts a list of `DataFrameRow` objects across
    that texts.
    Each DataFrameRow corresponds to a field, and each value in the row corresponds to
    a value for the field.

    Difference with DFEvaporateProgram is that 1) each DataFrameRow
    is column-oriented (instead of row-oriented), and 2)
    each DataFrameRow can be variable length (not guaranteed to have 1 value per
    node).

    """

    @classmethod
    def from_defaults(
        cls,
        fields_to_extract: Optional[List[str]] = None,
        fields_context: Optional[Dict[str, Any]] = None,
        service_context: Optional[ServiceContext] = None,
        schema_id_prompt: Optional[SchemaIDPrompt] = None,
        fn_generate_prompt: Optional[FnGeneratePrompt] = None,
        field_extract_query_tmpl: str = DEFAULT_FIELD_EXTRACT_QUERY_TMPL,
        nodes_to_fit: Optional[List[BaseNode]] = None,
        verbose: bool = False,
    ) -> "BaseEvaporateProgram":
        # modify the default function generate prompt to return a list
        fn_generate_prompt = fn_generate_prompt or FN_GENERATION_LIST_PROMPT
        return super().from_defaults(
            fields_to_extract=fields_to_extract,
            fields_context=fields_context,
            service_context=service_context,
            schema_id_prompt=schema_id_prompt,
            fn_generate_prompt=fn_generate_prompt,
            field_extract_query_tmpl=field_extract_query_tmpl,
            nodes_to_fit=nodes_to_fit,
            verbose=verbose,
        )

    def fit(
        self,
        nodes: List[BaseNode],
        field: str,
        field_context: Optional[Any] = None,
        expected_output: Optional[Any] = None,
        inplace: bool = True,
    ) -> str:
        """Given the input Nodes and fields, synthesize the python code."""
        fn = self._extractor.extract_fn_from_nodes(
            nodes, field, expected_output=expected_output
        )
        logger.debug(f"Extracted function: {fn}")
        if self._verbose:
            print_text(f"Extracted function: {fn}\n", color="blue")
        if inplace:
            self._field_fns[field] = fn
        return fn

    @property
    def output_cls(self) -> Type[DataFrameValuesPerColumn]:
        """Output class."""
        return DataFrameValuesPerColumn

    def _inference(
        self, nodes: List[BaseNode], fn_str: str, field_name: str
    ) -> List[Any]:
        """Given the input, call the python code and return the result."""
        results_by_node = self._extractor.run_fn_on_nodes(nodes, fn_str, field_name)
        # flatten results
        return [r for results in results_by_node for r in results]

    def __call__(self, *args: Any, **kwds: Any) -> DataFrameValuesPerColumn:
        """Call evaporate on inference data."""
        # TODO: either specify `nodes` or `texts` in kwds
        if "nodes" in kwds:
            nodes = kwds["nodes"]
        elif "texts" in kwds:
            nodes = [TextNode(text=t) for t in kwds["texts"]]
        else:
            raise ValueError("Must provide either `nodes` or `texts`.")

        col_dict = {}
        for field in self._fields:
            col_dict[field] = self._inference(nodes, self._field_fns[field], field)

        # convert col_dict to list of DataFrameRow objects
        df_row_objs = []
        for field in self._fields:
            df_row_objs.append(DataFrameRow(row_values=col_dict[field]))

        return DataFrameValuesPerColumn(columns=df_row_objs)
