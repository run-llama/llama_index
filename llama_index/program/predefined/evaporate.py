import logging
from typing import Any, Dict, List, Optional, Type, cast

from llama_index.experimental.evaporate.base import EvaporateExtractor
from llama_index.program.base_program import BasePydanticProgram
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.program.predefined.df import (
    DEFAULT_ROWS_DF_PARSER_TMPL,
    DataFrameRowsOnly,
)
from llama_index.prompts.prompts import Prompt
from llama_index.schema import BaseNode

logger = logging.getLogger(__name__)


class EvaporateProgram(BasePydanticProgram[DataFrameRowsOnly]):
    """Evaporate program. Given some fields you want to extract
    and some example text, synthesizes a python program to extract
    those values from the text.

    Requires
    """

    def __init__(
        self,
        pydantic_program_cls: Type[BaseLLMFunctionProgram],
        fields_to_extract: List[str],
        df_parser_template_str: str = DEFAULT_ROWS_DF_PARSER_TMPL,
        training_key: str = "training_data",
        infer_key: str = "infer_data",
        **program_kwargs: Any,
    ) -> None:
        """Init params."""
        # partial format df parser template string with column schema
        # NOTE: hack where we use prompt class to partial format
        orig_prompt = Prompt(df_parser_template_str)
        column_schema = ",".join(fields_to_extract)
        new_prompt = Prompt.from_prompt(
            orig_prompt.partial_format(
                column_schema=column_schema,
            )
        )

        pydantic_program = pydantic_program_cls.from_defaults(
            DataFrameRowsOnly, new_prompt.original_template, **program_kwargs
        )
        self._validate_program(pydantic_program)
        self._pydantic_program = pydantic_program
        self._extractor = EvaporateExtractor()
        self.field_fns: Dict[str, str] = {}
        self.fields = fields_to_extract
        self._training_key = training_key
        self._infer_key = infer_key

    def _validate_program(self, pydantic_program: BasePydanticProgram) -> None:
        if pydantic_program.output_cls != DataFrameRowsOnly:
            raise ValueError(
                "Output class of pydantic program must be `DataFramRowsOnly`."
            )

    def _fit(self, nodes: List[BaseNode], field: str) -> str:
        """Given the input Nodes and fields, synthesize the python code."""
        fn = self._extractor.extract_fn_from_nodes(nodes, field)
        logger.debug(f"Extracted function: {fn}")
        print(f"Extracted function: {fn}")
        return fn

    def _inference(self, nodes: List[BaseNode], fn_str: str, field_name: str) -> str:
        """Given the input, call the python code and return the result."""
        results = self._extractor.run_fn_on_nodes(nodes, fn_str, field_name)
        logger.debug(f"Results: {results}")
        print(f"Results: {results}")
        return str(results)

    @classmethod
    def from_defaults(
        cls,
        fields_to_extract: List[str],
        pydantic_program_cls: Optional[Type[BaseLLMFunctionProgram]] = None,
        df_parser_template_str: str = DEFAULT_ROWS_DF_PARSER_TMPL,
        **kwargs: Any,
    ) -> "EvaporateProgram":
        """Rows DF output parser."""
        pydantic_program_cls = pydantic_program_cls or OpenAIPydanticProgram

        return cls(
            pydantic_program_cls,
            fields_to_extract,
            df_parser_template_str=df_parser_template_str,
            **kwargs,
        )

    @property
    def output_cls(self) -> Type[DataFrameRowsOnly]:
        """Output class."""
        return DataFrameRowsOnly

    def __call__(self, *args: Any, **kwds: Any) -> DataFrameRowsOnly:
        """Call evaporate on training and inference data. Inputs should be two
        lists of nodes, using the keys `training_data` and `infer_data`."""
        for field in self.fields:
            self.field_fns[field] = self._fit(kwds[self._training_key], field)
        infer_dict = {}
        for field in self.fields:
            infer_dict[field] = self._inference(
                kwds[self._infer_key], self.field_fns[field], field
            )
        infer_results = str(infer_dict)
        result = self._pydantic_program(**{"input_str": infer_results})
        result = cast(DataFrameRowsOnly, result)
        return result
