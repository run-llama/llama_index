import os
from shutil import rmtree
from typing import Callable, Dict, List, Optional

import tqdm
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.utils import get_cache_dir


class BeirEvaluator:
    """
    Refer to: https://github.com/beir-cellar/beir for a full list of supported datasets
    and a full description of BEIR.
    """

    def __init__(self) -> None:
        try:
            pass
        except ImportError:
            raise ImportError(
                "Please install beir to use this feature: " "`pip install beir`",
            )

    def _download_datasets(self, datasets: List[str] = ["nfcorpus"]) -> Dict[str, str]:
        from beir import util

        cache_dir = get_cache_dir()

        dataset_paths = {}
        for dataset in datasets:
            dataset_full_path = os.path.join(cache_dir, "datasets", "BeIR__" + dataset)
            if not os.path.exists(dataset_full_path):
                url = f"""https://public.ukp.informatik.tu-darmstadt.de/thakur\
/BEIR/datasets/{dataset}.zip"""
                try:
                    util.download_and_unzip(url, dataset_full_path)
                except Exception as e:
                    print(
                        "Dataset:", dataset, "not found at:", url, "Removing cached dir"
                    )
                    rmtree(dataset_full_path)
                    raise ValueError(f"invalid BEIR dataset: {dataset}") from e

            print("Dataset:", dataset, "downloaded at:", dataset_full_path)
            dataset_paths[dataset] = os.path.join(dataset_full_path, dataset)
        return dataset_paths

    def run(
        self,
        create_retriever: Callable[[List[Document]], BaseRetriever],
        datasets: List[str] = ["nfcorpus"],
        metrics_k_values: List[int] = [3, 10],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ) -> None:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval

        dataset_paths = self._download_datasets(datasets)
        for dataset in datasets:
            dataset_path = dataset_paths[dataset]
            print("Evaluating on dataset:", dataset)
            print("-------------------------------------")

            corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
                split="test"
            )

            documents = []
            for id, val in corpus.items():
                doc = Document(
                    text=val["text"], metadata={"title": val["title"], "doc_id": id}
                )
                documents.append(doc)

            retriever = create_retriever(documents)

            print("Retriever created for: ", dataset)

            print("Evaluating retriever on questions against qrels")

            results = {}
            for key, query in tqdm.tqdm(queries.items()):
                nodes_with_score = retriever.retrieve(query)
                node_postprocessors = node_postprocessors or []
                for node_postprocessor in node_postprocessors:
                    nodes_with_score = node_postprocessor.postprocess_nodes(
                        nodes_with_score, query_bundle=QueryBundle(query_str=query)
                    )
                results[key] = {
                    node.node.metadata["doc_id"]: node.score
                    for node in nodes_with_score
                }

            ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(
                qrels, results, metrics_k_values
            )
            print("Results for:", dataset)
            for k in metrics_k_values:
                print(
                    {
                        f"NDCG@{k}": ndcg[f"NDCG@{k}"],
                        f"MAP@{k}": map_[f"MAP@{k}"],
                        f"Recall@{k}": recall[f"Recall@{k}"],
                        f"precision@{k}": precision[f"P@{k}"],
                    }
                )
            print("-------------------------------------")
