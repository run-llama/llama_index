from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from llama_index.core.response import Response
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.bridge.pydantic import Field
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.utils import print_text

_IMPORT_ERROR_MSG = (
    "`llama_cpp` package not found, please run `pip install llama_cpp_python`"
)

_RELEVANCE_TOKENS = ["[Irrelevant]", "[Relevant]"]

_RETRIEVAL_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
_UTILITY_TOKENS = [
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]
_GROUND_TOKENS = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
]
_CTRL_TOKENS = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
    "[No Retrieval]",
    "[Retrieval]",
    "[Irrelevant]",
    "[Relevant]",
    "[Continue to Use Evidence]",
    "<paragraph>",
    "</paragraph>",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]

_MODEL_KWARGS = {"logits_all": True, "n_ctx": 2048, "n_gpu_layers": -1}
_GENERATE_KWARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 50,
    "logprobs": 32016,
}


@dataclass
class CriticOutput:
    llm_response_per_paragraph: Dict[int, str]
    paragraphs_final_score: Dict[int, float]
    source_nodes: List[NodeWithScore]


def _format_prompt(input: str, paragraph: str = None) -> str:
    prompt = f"### Instruction:\n{input}\n\n### Response:\n"
    if paragraph is not None:
        prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
    return prompt


def _postprocess_answer(answer: str) -> str:
    for token in _CTRL_TOKENS:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


def _relevance_score(pred_log_probs: Dict[str, float]) -> float:
    """
    Compute relevance score.

    Args:
        pred_log_probs (Dict[str, float]): log probabilities of tokens

    Returns:
        float: relevance score

    """
    rel_prob = np.exp(float(pred_log_probs["[Relevant]"]))
    irel_prob = np.exp(float(pred_log_probs["[Irrelevant]"]))
    return rel_prob / (rel_prob + irel_prob)


def _is_supported_score(
    pred_tokens: List[int], pred_log_probs_dict: List[Dict[str, float]]
) -> float:
    """
    Compute support score.

    Args:
        pred_tokens (List[int]): List of predicted tokens
        pred_log_probs_dict (List[Dict[str, float]]): log probabilities of tokens for each predicted tokens

    Returns:
        float: support score

    """
    isSup_score = 0
    groundness_token_appear_id = -1
    for tok_idx, token in enumerate(pred_tokens):
        if token in _GROUND_TOKENS:
            groundness_token_appear_id = tok_idx
            break
    if groundness_token_appear_id > -1:
        grd_score_dict = {}
        for token in _GROUND_TOKENS:
            prob = pred_log_probs_dict[groundness_token_appear_id][token]
            grd_score_dict[token] = np.exp(float(prob))
        isSup_score = (
            grd_score_dict["[Fully supported]"]
            + 0.5 * grd_score_dict["[Partially supported]"]
        ) / np.sum(list(grd_score_dict.values()))
    return isSup_score


def _is_useful_score(
    pred_tokens: List[int], pred_log_probs_dict: List[Dict[str, float]]
) -> float:
    """
    Compute usefulness score.

    Args:
        pred_tokens (List[int]): List of predicted tokens
        pred_log_probs_dict (List[Dict[str, float]]): log probabilities of tokens for each predicted tokens

    Returns:
        float: relevance score

    """
    isUse_score = 0
    utility_token_appear_id = -1
    for tok_idx, tok in enumerate(pred_tokens):
        if tok in _UTILITY_TOKENS:
            utility_token_appear_id = tok_idx
    if utility_token_appear_id > -1:
        ut_score_dict = {}
        for token in _UTILITY_TOKENS:
            prob = pred_log_probs_dict[utility_token_appear_id][token]
            ut_score_dict[token] = np.exp(float(prob))

        ut_sum = np.sum(list(ut_score_dict.values()))
        ut_weights = [-1, -0.5, 0, 0.5, 1]
        isUse_score = np.sum(
            [
                ut_weights[i] * (ut_score_dict[f"[Utility:{i + 1}]"] / ut_sum)
                for i in range(len(ut_weights))
            ]
        )
    return isUse_score


class SelfRAGQueryEngine(CustomQueryEngine):
    """Simple short form self RAG query engine."""

    llm: Any = Field(default=None, description="llm")
    retriever: BaseRetriever = Field(default=None, description="retriever")
    generate_kwargs: Dict = Field(default=None, description="llm generation arguments")
    verbose: bool = Field(default=True, description="Verbose.")

    def __init__(
        self,
        model_path: str,
        retriever: BaseRetriever,
        verbose: bool = False,
        model_kwargs: Dict = None,
        generate_kwargs: Dict = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(verbose=verbose, **kwargs)
        model_kwargs = model_kwargs or _MODEL_KWARGS
        self.generate_kwargs = generate_kwargs or _GENERATE_KWARGS
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(_IMPORT_ERROR_MSG)
        self.llm = Llama(model_path=model_path, verbose=verbose, **model_kwargs)
        self.retriever = retriever

    def _run_critic(self, paragraphs: List[str]) -> CriticOutput:
        """
        Run Critic component, the llm will generate responses based on the paragraphs and then evaluate them.

        Args:
            paragraphs (List[str]): List of paragraphs to evaluate

        Returns:
            CriticOutput: Paragraphs final score, LLM predictions and source nodes

        """
        paragraphs_final_score = {}
        llm_response_text = {}
        source_nodes = []

        for p_idx, paragraph in enumerate(paragraphs):
            pred = self.llm(paragraph, **self.generate_kwargs)
            # Cache llm answer
            llm_response_text[p_idx] = pred["choices"][0]["text"]

            logprobs = pred["choices"][0]["logprobs"]
            pred_log_probs = logprobs["top_logprobs"]
            # Compute isRel score, on the first predicted token
            isRel_score = _relevance_score(pred_log_probs[0])

            # Compute isSup score
            isSup_score = _is_supported_score(logprobs["tokens"], pred_log_probs)

            # Compute isUse score
            isUse_score = _is_useful_score(logprobs["tokens"], pred_log_probs)

            paragraphs_final_score[p_idx] = (
                isRel_score + isSup_score + 0.5 * isUse_score
            )
            # Add the paragraph as source node with its relevance score
            source_nodes.append(
                NodeWithScore(
                    node=TextNode(text=paragraph, id_=str(p_idx)),
                    score=isRel_score,
                )
            )

            if self.verbose:
                print_text(
                    f"Input: {paragraph}\nPrediction: {llm_response_text[p_idx]}\nScore: {paragraphs_final_score[p_idx]}\n",
                    color="blue",
                )
                print_text(
                    f"{p_idx + 1}/{len(paragraphs)} paragraphs done\n\n", color="blue"
                )

        return CriticOutput(llm_response_text, paragraphs_final_score, source_nodes)

    def custom_query(self, query_str: str) -> Response:
        """Run self-RAG."""
        response = self.llm(prompt=_format_prompt(query_str), **_GENERATE_KWARGS)
        answer = response["choices"][0]["text"]
        source_nodes = []

        if "[Retrieval]" in answer:
            if self.verbose:
                print_text("Retrieval required\n", color="blue")
            documents = self.retriever.retrieve(query_str)
            if self.verbose:
                print_text(f"Received: {len(documents)} documents\n", color="blue")
            paragraphs = [
                _format_prompt(query_str, document.node.text) for document in documents
            ]

            if self.verbose:
                print_text("Start evaluation\n", color="blue")

            critic_output = self._run_critic(paragraphs)

            paragraphs_final_score = critic_output.paragraphs_final_score
            llm_response_per_paragraph = critic_output.llm_response_per_paragraph
            source_nodes = critic_output.source_nodes

            if self.verbose:
                print_text("End evaluation\n", color="blue")

            best_paragraph_id = max(
                paragraphs_final_score, key=paragraphs_final_score.get
            )
            answer = llm_response_per_paragraph[best_paragraph_id]
            if self.verbose:
                print_text(f"Selected the best answer: {answer}\n", color="blue")

        answer = _postprocess_answer(answer)
        if self.verbose:
            print_text(f"Final answer: {answer}\n", color="green")
        return Response(response=str(answer), source_nodes=source_nodes)


class SelfRAGPack(BaseLlamaPack):
    """Simple short form Self-RAG pack."""

    def __init__(
        self,
        model_path: str,
        retriever: BaseRetriever,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.query_engine = SelfRAGQueryEngine(model_path, retriever, verbose)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "query_engine": self.query_engine,
            "llm": self.query_engine.llm,
            "retriever": self.query_engine.retriever,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
