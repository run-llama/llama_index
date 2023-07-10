from typing import Callable, List, Dict
from llama_index.utils import get_cache_dir
from llama_index.schema import Document
from llama_index.indices.base_retriever import BaseRetriever
import os
import tqdm

beir_datasets = [
    "trec-covid",
    "hotpotqa",
]


class BeirEvaluator:
    def __init__(self) -> None:
        try:
            import beir
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
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
                util.download_and_unzip(url, dataset_full_path)
            print("Dataset:", dataset, "downloaded at:", dataset_full_path)
            dataset_paths[dataset] = os.path.join(dataset_full_path, dataset)
        return dataset_paths

    def run(
        self,
        create_retriever: Callable[[List[Document]], BaseRetriever],
        datasets: List[str] = ["nfcorpus"],
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
                documents.append(
                    Document(text=val["text"], metadata={"title": val["title"]}, id_=id)
                )

            retriever = create_retriever(documents)

            print("Retriever created for: ", dataset)

            results = {}
            for key, query in tqdm.tqdm(queries.items()):
                nodes_with_score = retriever.retrieve(query)
                results[key] = {
                    node.node.node_id: node.score for node in nodes_with_score
                }

            ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(
                qrels, results, [1, 10, 100]
            )
            print("Results for:", dataset)
            print(
                {
                    "NDCG@10": ndcg["NDCG@10"],
                    "Recall@100": recall["Recall@100"],
                    "precision": precision,
                }
            )
            print("-------------------------------------")
