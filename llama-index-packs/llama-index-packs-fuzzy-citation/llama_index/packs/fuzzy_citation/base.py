from collections import defaultdict
from typing import Any, Dict

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.query_engine import BaseQueryEngine, CustomQueryEngine
from llama_index.core.schema import MetadataMode

DEFAULT_THRESHOLD = 50


class FuzzyCitationQueryEngine(CustomQueryEngine):
    """
    Runs any query engine and then analyzes the response to find relevant sentences.

    Using fuzzy matching, response.metadata is assigned to a dictionary containing a
    mapping of response+node sentence pairs and the node text start and end character
    indices in the original document + the node that the sentence came from.

    Example:
      print(response.metadata)
      >>> {("response_sent", "node_sent"): (0, 25, <BaseNode>), ...}

    """

    query_engine: BaseQueryEngine
    threshold: int = DEFAULT_THRESHOLD

    def get_relevant_sentences(
        self, response: RESPONSE_TYPE, threshold: int = DEFAULT_THRESHOLD
    ) -> Dict[str, Dict[str, Any]]:
        """Get relevant sentences from a response."""
        from thefuzz import fuzz

        tokenizer = split_by_sentence_tokenizer()

        results = {}
        for source_node in response.source_nodes:
            node = source_node.node
            response_sentences = tokenizer(str(response))
            node_sentences = tokenizer(
                node.get_content(metadata_mode=MetadataMode.NONE)
            )

            result_matrix = {}
            for i, response_sentence in enumerate(response_sentences):
                for j, node_sentence in enumerate(node_sentences):
                    result_matrix[(i, j)] = fuzz.ratio(response_sentence, node_sentence)

            top_sentences = {}
            for j in range(len(node_sentences)):
                scores = [result_matrix[(i, j)] for i in range(len(response_sentences))]
                max_value = max(scores)
                if max_value > threshold:
                    response_sent_idx = scores.index(max_value)
                    top_sentences[(response_sent_idx, j)] = node_sentences[j]

            # concat nearby sentences
            top_chunks = defaultdict(list)
            prev_idx = -1
            for response_sent_idx, node_sent_idx in sorted(top_sentences.keys()):
                if prev_idx == -1:
                    top_chunks[response_sent_idx].append(
                        top_sentences[(response_sent_idx, node_sent_idx)]
                    )
                elif node_sent_idx - prev_idx == 1:
                    top_chunks[response_sent_idx][-1] += top_sentences[
                        (response_sent_idx, node_sent_idx)
                    ]
                else:
                    top_chunks[response_sent_idx].append(
                        top_sentences[(response_sent_idx, node_sent_idx)]
                    )
                prev_idx = node_sent_idx

            # associate chunks with their nodes
            for response_sent_idx, chunks in top_chunks.items():
                for chunk in chunks:
                    start_char_idx = node.get_content(
                        metadata_mode=MetadataMode.NONE
                    ).find(chunk)
                    end_char_idx = start_char_idx + len(chunk)

                    response_sent = response_sentences[response_sent_idx]
                    results[(response_sent, chunk)] = {
                        "start_char_idx": start_char_idx,
                        "end_char_idx": end_char_idx,
                        "node": node,
                    }

        return results

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        """Custom query."""
        response = self.query_engine.query(query_str)
        fuzzy_citations = self.get_relevant_sentences(
            response, threshold=self.threshold
        )
        response.metadata = fuzzy_citations
        return response

    async def acustom_query(self, query_str: str) -> RESPONSE_TYPE:
        """Async custom query."""
        response = await self.query_engine.aquery(query_str)
        fuzzy_citations = self.get_relevant_sentences(
            response, threshold=self.threshold
        )
        response.metadata = fuzzy_citations
        return response


class FuzzyCitationEnginePack(BaseLlamaPack):
    def __init__(
        self, query_engine: BaseQueryEngine, threshold: int = DEFAULT_THRESHOLD
    ) -> None:
        """Init params."""
        try:
            from thefuzz import fuzz  # noqa: F401
        except ImportError:
            raise ImportError(
                "Please run `pip install thefuzz` to use the fuzzy citation engine."
            )

        self.query_engine = FuzzyCitationQueryEngine(
            query_engine=query_engine, threshold=threshold
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "query_engine": self.query_engine,
            "query_engine_cls": FuzzyCitationQueryEngine,
        }

    def run(self, query_str: str, **kwargs: Any) -> RESPONSE_TYPE:
        """Run the pipeline."""
        return self.query_engine.query(query_str)
