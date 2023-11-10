import json
import os
import re
import string
from collections import Counter
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple

import requests
import tqdm

from llama_index.core import BaseQueryEngine, BaseRetriever
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.utils import get_cache_dir

DEV_DISTRACTOR_URL = """http://curtis.ml.cmu.edu/datasets/\
hotpot/hotpot_dev_distractor_v1.json"""


class HotpotQAEvaluator:
    """
    Refer to https://hotpotqa.github.io/ for more details on the dataset.
    """

    def _download_datasets(self) -> Dict[str, str]:
        cache_dir = get_cache_dir()

        dataset_paths = {}
        dataset = "hotpot_dev_distractor"
        dataset_full_path = os.path.join(cache_dir, "datasets", "HotpotQA")
        if not os.path.exists(dataset_full_path):
            url = DEV_DISTRACTOR_URL
            try:
                os.makedirs(dataset_full_path, exist_ok=True)
                save_file = open(
                    os.path.join(dataset_full_path, "dev_distractor.json"), "wb"
                )
                response = requests.get(url, stream=True)

                # Define the size of each chunk
                chunk_size = 1024

                # Loop over the chunks and parse the JSON data
                for chunk in tqdm.tqdm(response.iter_content(chunk_size=chunk_size)):
                    if chunk:
                        save_file.write(chunk)
            except Exception as e:
                if os.path.exists(dataset_full_path):
                    print(
                        "Dataset:", dataset, "not found at:", url, "Removing cached dir"
                    )
                    rmtree(dataset_full_path)
                raise ValueError(f"could not download {dataset} dataset") from e
        dataset_paths[dataset] = os.path.join(dataset_full_path, "dev_distractor.json")
        print("Dataset:", dataset, "downloaded at:", dataset_full_path)
        return dataset_paths

    def run(
        self,
        query_engine: BaseQueryEngine,
        queries: int = 10,
        queries_fraction: Optional[float] = None,
        show_result: bool = False,
    ) -> None:
        dataset_paths = self._download_datasets()
        dataset = "hotpot_dev_distractor"
        dataset_path = dataset_paths[dataset]
        print("Evaluating on dataset:", dataset)
        print("-------------------------------------")

        f = open(dataset_path)
        query_objects = json.loads(f.read())
        if queries_fraction:
            queries_to_load = int(len(query_objects) * queries_fraction)
        else:
            queries_to_load = queries
            queries_fraction = round(queries / len(query_objects), 5)

        print(
            f"Loading {queries_to_load} queries out of \
{len(query_objects)} (fraction: {queries_fraction})"
        )
        query_objects = query_objects[:queries_to_load]

        assert isinstance(
            query_engine, RetrieverQueryEngine
        ), "query_engine must be a RetrieverQueryEngine for this evaluation"
        retriever = HotpotQARetriever(query_objects)
        # Mock the query engine with a retriever
        query_engine = query_engine.with_retriever(retriever=retriever)

        scores = {"exact_match": 0.0, "f1": 0.0}

        for query in query_objects:
            query_bundle = QueryBundle(
                query_str=query["question"]
                + " Give a short factoid answer (as few words as possible).",
                custom_embedding_strs=[query["question"]],
            )
            response = query_engine.query(query_bundle)
            em = int(
                exact_match_score(
                    prediction=str(response), ground_truth=query["answer"]
                )
            )
            f1, _, _ = f1_score(prediction=str(response), ground_truth=query["answer"])
            scores["exact_match"] += em
            scores["f1"] += f1
            if show_result:
                print("Question: ", query["question"])
                print("Response:", response)
                print("Correct answer: ", query["answer"])
                print("EM:", em, "F1:", f1)
                print("-------------------------------------")

        for score in scores:
            scores[score] /= len(query_objects)

        print("Scores: ", scores)


class HotpotQARetriever(BaseRetriever):
    """
    This is a mocked retriever for HotpotQA dataset. It is only meant to be used
    with the hotpotqa dev dataset in the distractor setting. This is the setting that
    does not require retrieval but requires identifying the supporting facts from
    a list of 10 sources.
    """

    def __init__(self, query_objects: Any) -> None:
        assert isinstance(
            query_objects,
            list,
        ), f"query_objects must be a list, got: {type(query_objects)}"
        self._queries = {}
        for object in query_objects:
            self._queries[object["question"]] = object

    def _retrieve(self, query: QueryBundle) -> List[NodeWithScore]:
        if query.custom_embedding_strs:
            query_str = query.custom_embedding_strs[0]
        else:
            query_str = query.query_str
        contexts = self._queries[query_str]["context"]
        node_with_scores = []
        for ctx in contexts:
            text_list = ctx[1]
            text = "\n".join(text_list)
            node = TextNode(text=text, metadata={"title": ctx[0]})
            node_with_scores.append(NodeWithScore(node=node, score=1.0))

        return node_with_scores

    def __str__(self) -> str:
        return "HotpotQARetriever"


"""
Utils from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)
