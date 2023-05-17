"""Utilities for GPT indices."""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from llama_index.data_structs.node import Node
from llama_index.utils import globals_helper, truncate_text
from llama_index.vector_stores.types import VectorStoreQueryResult

_logger = logging.getLogger(__name__)


def get_sorted_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """Get sorted node list. Used by tree-strutured indices."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def extract_numbers_given_response(response: str, n: int = 1) -> Optional[List[int]]:
    """Extract number given the GPT-generated response.

    Used by tree-structured indices.

    """
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return numbers[:n]


def expand_tokens_with_subtokens(tokens: Set[str]) -> Set[str]:
    """Get subtokens from a list of tokens., filtering for stopwords."""
    results = set()
    for token in tokens:
        results.add(token)
        sub_tokens = re.findall(r"\w+", token)
        if len(sub_tokens) > 1:
            results.update({w for w in sub_tokens if w not in globals_helper.stopwords})

    return results


def log_vector_store_query_result(
    result: VectorStoreQueryResult, logger: Optional[logging.Logger] = None
) -> None:
    """Log vector store query result."""
    logger = logger or _logger

    assert result.ids is not None
    assert result.nodes is not None
    similarities = (
        result.similarities
        if result.similarities is not None and len(result.similarities) > 0
        else [1.0 for _ in result.ids]
    )

    fmt_txts = []
    for node_idx, node_similarity, node in zip(result.ids, similarities, result.nodes):
        fmt_txt = f"> [Node {node_idx}] [Similarity score: \
            {float(node_similarity):.6}] {truncate_text(node.get_text(), 100)}"
        fmt_txts.append(fmt_txt)
    top_k_node_text = "\n".join(fmt_txts)
    logger.debug(f"> Top {len(result.nodes)} nodes:\n{top_k_node_text}")


def default_format_node_batch_fn(
    summary_nodes: List[Node],
) -> str:
    """Default format node batch function.

    Assign each summary node a number, and format the batch of nodes.

    """
    fmt_node_txts = []
    for idx in range(len(summary_nodes)):
        number = idx + 1
        fmt_node_txts.append(f"Document {number}:\n{summary_nodes[idx].get_text()}")
    return "\n\n".join(fmt_node_txts)


def default_parse_choice_select_answer_fn(
    answer: str, num_choices: int, raise_error: bool = False
) -> Tuple[List[int], Optional[List[float]]]:
    """Default parse choice select answer function."""
    answer_lines = answer.split("\n")
    answer_nums = []
    answer_relevances = []
    for answer_line in answer_lines:
        line_tokens = answer_line.split(",")
        if len(line_tokens) != 2:
            if not raise_error:
                continue
            else:
                raise ValueError(
                    f"Invalid answer line: {answer_line}. "
                    "Answer line must be of the form: "
                    "answer_num: <int>, answer_relevance: <float>"
                )
        answer_num = int(line_tokens[0].split(":")[1].strip())
        if answer_num > num_choices:
            continue
        answer_nums.append(answer_num)
        answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))
    return answer_nums, answer_relevances
