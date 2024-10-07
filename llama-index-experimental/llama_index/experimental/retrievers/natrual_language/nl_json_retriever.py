from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.llms import llm
from llama_index.core.prompts import BasePromptTemplate
from typing import Optional

import logging
import pandas as pd
import json

from llama_index.experimental.retrievers.natrual_language import NLDataframeRetriever

logger = logging.getLogger(__name__)


class NLJsonRetriever(NLDataframeRetriever):
    def __init__(
        self,
        json_path: str,
        llm: llm,
        name: Optional[str] = None,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        data = json.loads(open(json_path).read())
        df = pd.DataFrame.from_dict(data, orient="columns")

        super().__init__(
            df=df,
            llm=llm,
            text_to_sql_prompt=text_to_sql_prompt,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            name=name,
            callback_manager=callback_manager,
        )
