"""Node recency post-processor."""

from llama_index.indices.postprocessor.node import BasePydanticNodePostprocessor
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.data_structs.node import NodeWithScore
from pydantic import Field, validator
from typing import Optional, List, Set
import pandas as pd
import numpy as np
from datetime import datetime
from llama_index.storage.storage_context import StorageContext

# NOTE: currently not being used
# DEFAULT_INFER_RECENCY_TMPL = (
#     "A question is provided.\n"
#     "The goal is to determine whether the question requires finding the most recent "
#     "context.\n"
#     "Please respond with YES or NO.\n"
#     "Question: What is the current status of the patient?\n"
#     "Answer: YES\n"
#     "Question: What happened in the Battle of Yorktown?\n"
#     "Answer: NO\n"
#     "Question: What are the most recent changes to the project?\n"
#     "Answer: YES\n"
#     "Question: How did Harry defeat Voldemort in the Battle of Hogwarts?\n"
#     "Answer: NO\n"
#     "Question: {query_str}\n"
#     "Answer: "
# )


# def parse_recency_pred(pred: str) -> bool:
#     """Parse recency prediction."""
#     if "YES" in pred:
#         return True
#     elif "NO" in pred:
#         return False
#     else:
#         raise ValueError(f"Invalid recency prediction: {pred}.")


class FixedRecencyPostprocessor(BasePydanticNodePostprocessor):
    """Recency post-processor.

    This post-processor does the following steps:

    - Decides if we need to use the post-processor given the query
      (is it temporal-related?)
    - If yes, sorts nodes by date.
    - Take the first k nodes (by default 1), and use that to synthesize an answer.

    """

    service_context: ServiceContext
    top_k: int = 1
    # infer_recency_tmpl: str = Field(default=DEFAULT_INFER_RECENCY_TMPL)
    date_key: str = "date"
    # if false, then search node info
    in_extra_info: bool = True

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        # query_bundle = cast(QueryBundle, extra_info["query_bundle"])
        # infer_recency_prompt = SimpleInputPrompt(self.infer_recency_tmpl)
        # raw_pred, _ = self.service_context.llm_predictor.predict(
        #     prompt=infer_recency_prompt,
        #     query_str=query_bundle.query_str,
        # )
        # pred = parse_recency_pred(raw_pred)
        # # if no need to use recency post-processor, return nodes as is
        # if not pred:
        #     return nodes

        # sort nodes by date
        info_dict_attr = "extra_info" if self.in_extra_info else "node_info"
        node_dates = pd.to_datetime(
            [getattr(node.node, info_dict_attr)[self.date_key] for node in nodes]
        )
        sorted_node_idxs = np.flip(node_dates.argsort())
        sorted_nodes = [nodes[idx] for idx in sorted_node_idxs]

        return sorted_nodes[: self.top_k]


DEFAULT_QUERY_EMBEDDING_TMPL = (
    "The current document is provided.\n"
    "----------------\n"
    "{context_str}\n"
    "----------------\n"
    "Given the document, we wish to find documents that contain \n"
    "similar context. Note that these documents are older "
    "than the current document, meaning that certain details may be changed. \n"
    "However, the high-level context should be similar.\n"
)


class EmbeddingRecencyPostprocessor(BasePydanticNodePostprocessor):
    """Recency post-processor.

    This post-processor does the following steps:

    - Decides if we need to use the post-processor given the query
      (is it temporal-related?)
    - If yes, sorts nodes by date.
    - 'embedding_filter_level' can be set to 'nodes' or 'documents'. Which chooses
      whether to use the node or document embeddings to filter nodes.

      If 'nodes', filter out nodes that have high embedding similarity with more
      recent nodes.

      If 'documents', get the doc embedding by averaging all the node embeddings
      for the document, then filter out documents that have high embedding similarity
      with more recent documents.
    """

    service_context: ServiceContext
    storage_context: Optional[StorageContext] = None
    # infer_recency_tmpl: str = Field(default=DEFAULT_INFER_RECENCY_TMPL)
    date_key: str = "date"
    # if false, then search node info
    in_extra_info: bool = True
    similarity_cutoff: float = Field(default=0.7)
    query_embedding_tmpl: str = Field(default=DEFAULT_QUERY_EMBEDDING_TMPL)
    embedding_filter_level: str = Field(default="nodes")

    """
    options are 'nodes' or 'documents'
        nodes -> get node embeddings and dedup very similar nodes, keep the most recent
        documents -> get doc embedding by averaging node embeddings in the doc
        and dedup very similar docs, keep the most recent
    """

    @validator("embedding_filter_level")
    def validate_embedding_filter_level(cls, value: str) -> str:
        allowed_values = ["nodes", "documents"]
        if value not in allowed_values:
            raise ValueError(f"embedding_filter_level must be one of {allowed_values}")
        return value

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        # sort nodes by date in asc order
        info_dict_attr = "extra_info" if self.in_extra_info else "node_info"
        node_dates = pd.to_datetime(
            [getattr(node.node, info_dict_attr)[self.date_key] for node in nodes]
        )
        sorted_node_idxs = np.flip(node_dates.argsort())
        sorted_nodes: List[NodeWithScore] = [nodes[idx] for idx in sorted_node_idxs]

        if self.embedding_filter_level == "documents":
            if self.storage_context is None:
                raise ValueError("Missing storage context.")
            # get the node_ids for each retrieved document
            sorted_doc_ids = list(
                dict.fromkeys(
                    [
                        node.node.ref_doc_id
                        for node in sorted_nodes
                        if node.node.ref_doc_id is not None
                    ]
                )
            )
            sorted_node_ids_for_docs = []
            for ref_doc_id in sorted_doc_ids:
                doc_info = self.storage_context.docstore.get_ref_doc_info(ref_doc_id)
                if doc_info is not None and doc_info.doc_ids is not None:
                    sorted_node_ids_for_docs.append(doc_info.doc_ids)
                else:
                    sorted_node_ids_for_docs.append([])

            # get the embeddings for each node in each retrieved document
            embed_model = self.service_context.embed_model
            for node_ids in sorted_node_ids_for_docs:
                for node_id in node_ids:
                    _node = self.storage_context.docstore.get_document(node_id)
                    if _node is not None:
                        embed_model.queue_text_for_embedding(node_id, _node.get_text())

            # get the embeddings for each doc by averaging the doc's node embeddings
            _, text_embeddings = embed_model.get_queued_text_embeddings()
            idx_offset = 0
            sorted_doc_embeddings = []
            for node_ids in sorted_node_ids_for_docs:
                doc_embedding = np.mean(
                    text_embeddings[idx_offset : idx_offset + len(node_ids)], axis=0
                )
                sorted_doc_embeddings.append(doc_embedding)
                idx_offset += len(node_ids)

            # calculate cosine similarity for each pair of document vectors
            doc_ids_to_skip: Set[str] = set()
            num_of_embeddings = len(sorted_doc_embeddings)
            for i in range(num_of_embeddings):
                for j in range(i + 1, num_of_embeddings):
                    if sorted_doc_ids[j] in doc_ids_to_skip:
                        continue
                    cosine_sim = np.dot(
                        sorted_doc_embeddings[i], sorted_doc_embeddings[j]
                    )
                    older_doc_id = sorted_doc_ids[j]
                    if cosine_sim > self.similarity_cutoff and older_doc_id:
                        doc_ids_to_skip.add(older_doc_id)

            # return filtered nodes
            filtered_nodes = [
                node
                for node in sorted_nodes
                if node.node.ref_doc_id not in doc_ids_to_skip
            ]

            return filtered_nodes
        else:
            # get embeddings for each node
            embed_model = self.service_context.embed_model
            for node in sorted_nodes:
                embed_model.queue_text_for_embedding(
                    node.node.get_doc_id(), node.node.get_text()
                )

            _, text_embeddings = embed_model.get_queued_text_embeddings()
            node_ids_to_skip: Set[str] = set()
            for idx, node in enumerate(sorted_nodes):
                if node.node.get_doc_id() in node_ids_to_skip:
                    continue
                # get query embedding for the "query" node
                # NOTE: not the same as the text embedding because
                # we want to optimize for retrieval results

                query_text = self.query_embedding_tmpl.format(
                    context_str=node.node.get_text(),
                )
                query_embedding = embed_model.get_query_embedding(query_text)

                for idx2 in range(idx + 1, len(sorted_nodes)):
                    if sorted_nodes[idx2].node.get_doc_id() in node_ids_to_skip:
                        continue
                    node2 = sorted_nodes[idx2]
                    cosine_sim = np.dot(query_embedding, text_embeddings[idx2])
                    if cosine_sim > self.similarity_cutoff:
                        node_ids_to_skip.add(node2.node.get_doc_id())
            filtered_nodes = [
                node
                for node in sorted_nodes
                if node.node.get_doc_id() not in node_ids_to_skip
            ]
            return filtered_nodes


class TimeWeightedPostprocessor(BasePydanticNodePostprocessor):
    """Time-weighted post-processor.

    Reranks a set of nodes based on their recency.

    """

    time_decay: float = Field(default=0.99)
    last_accessed_key: str = "__last_accessed__"
    time_access_refresh: bool = True
    # optionally set now (makes it easier to test)
    now: Optional[float] = None
    top_k: int = 1

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        now = self.now or datetime.now().timestamp()
        # TODO: refactor with get_top_k_embeddings

        similarities = []
        for node_with_score in nodes:
            # embedding similarity score
            score = node_with_score.score or 1.0
            node = node_with_score.node
            # time score
            if node.node_info is None:
                raise ValueError("node_info is None")

            last_accessed = node.node_info.get(self.last_accessed_key, None)
            if last_accessed is None:
                last_accessed = now

            hours_passed = (now - last_accessed) / 3600
            time_similarity = (1 - self.time_decay) ** hours_passed

            similarity = score + time_similarity

            similarities.append(similarity)

        sorted_tups = sorted(zip(similarities, nodes), key=lambda x: x[0], reverse=True)

        top_k = min(self.top_k, len(sorted_tups))
        result_tups = sorted_tups[:top_k]
        result_nodes = [NodeWithScore(n.node, score) for score, n in result_tups]

        # set __last_accessed__ to now
        if self.time_access_refresh:
            for node_with_score in result_nodes:
                node_with_score.node.get_node_info()[self.last_accessed_key] = now

        return result_nodes
