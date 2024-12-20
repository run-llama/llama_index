"""
Minorly tweaked from https://github.com/parthsarthi03/raptor/blob/master/raptor/cluster_tree_builder.py.

Full credits to the original authors!
"""

import numpy as np
import random
import tiktoken
import umap
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Optional

from llama_index.core.schema import BaseNode


# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    # If the number of embeddings is less than or equal to the dimension, return a list of zeros
    # This means all nodes are in the same cluster.
    # Otherwise, we will get an error when trying to cluster.
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def get_clusters(
    nodes: List[BaseNode],
    embedding_map: Dict[str, List[List[float]]],
    max_length_in_cluster: int = 10000,  # 10k tokens max per cluster
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base"),
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length=None,  # to keep track of the total length of the previous clusters
) -> List[List[BaseNode]]:
    # get embeddings
    embeddings = np.array([np.array(embedding_map[node.id_]) for node in nodes])

    # Perform the clustering
    clusters = perform_clustering(
        embeddings, dim=reduction_dimension, threshold=threshold
    )

    # Initialize an empty list to store the clusters of nodes
    node_clusters = []

    # Iterate over each unique label in the clusters
    for label in np.unique(np.concatenate(clusters)):
        # Get the indices of the nodes that belong to this cluster
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]

        # Add the corresponding nodes to the node_clusters list
        cluster_nodes = [nodes[i] for i in indices]

        # Base case: if the cluster only has one node, do not attempt to recluster it
        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        # Calculate the total length of the text in the nodes
        total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])

        # If the total length exceeds the maximum allowed length, recluster this cluster
        # If the total length did not change from the previous call then don't try again to avoid infinite recursion!
        if total_length > max_length_in_cluster and (
            prev_total_length is None or total_length < prev_total_length
        ):
            node_clusters.extend(
                get_clusters(
                    cluster_nodes,
                    embedding_map,
                    max_length_in_cluster=max_length_in_cluster,
                    tokenizer=tokenizer,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    prev_total_length=total_length,
                )
            )
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters
