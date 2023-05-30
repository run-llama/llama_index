"""GPT Tree Index inserter."""

from typing import Optional, Sequence

from llama_index.data_structs.data_structs import IndexGraph
from llama_index.data_structs.node import Node
from llama_index.indices.tree.utils import get_numbered_text_from_nodes
from llama_index.storage.docstore import BaseDocumentStore
from llama_index.storage.docstore.registry import get_default_docstore
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import (
    extract_numbers_given_response,
    get_sorted_node_list,
)
from llama_index.prompts.base import Prompt
from llama_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)


class GPTTreeIndexInserter:
    """LlamaIndex inserter."""

    def __init__(
        self,
        index_graph: IndexGraph,
        service_context: ServiceContext,
        num_children: int = 10,
        insert_prompt: Prompt = DEFAULT_INSERT_PROMPT,
        summary_prompt: Prompt = DEFAULT_SUMMARY_PROMPT,
        docstore: Optional[BaseDocumentStore] = None,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self.insert_prompt = insert_prompt
        self.index_graph = index_graph
        self._service_context = service_context
        self._docstore = docstore or get_default_docstore()

    def _insert_under_parent_and_consolidate(
        self, text_node: Node, parent_node: Optional[Node]
    ) -> None:
        """Insert node under parent and consolidate.

        Consolidation will happen by dividing up child nodes, and creating a new
        intermediate layer of nodes.

        """
        # perform insertion
        self.index_graph.insert_under_parent(text_node, parent_node)

        # if under num_children limit, then we're fine
        if len(self.index_graph.get_children(parent_node)) <= self.num_children:
            return
        else:
            # perform consolidation
            cur_graph_node_ids = self.index_graph.get_children(parent_node)
            cur_graph_nodes = self._docstore.get_node_dict(cur_graph_node_ids)
            cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
            # this layer is all leaf nodes, consolidate and split leaf nodes
            # consolidate and split leaf nodes in half
            # TODO: do better splitting (with a GPT prompt etc.)
            half1 = cur_graph_node_list[: len(cur_graph_nodes) // 2]
            half2 = cur_graph_node_list[len(cur_graph_nodes) // 2 :]

            truncated_chunks = self._service_context.prompt_helper.truncate(
                prompt=self.summary_prompt,
                text_chunks=[node.get_text() for node in half1],
            )
            text_chunk1 = "\n".join(truncated_chunks)

            summary1, _ = self._service_context.llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk1
            )
            node1 = Node(
                text=summary1,
            )
            self.index_graph.insert(node1, children_nodes=half1)

            truncated_chunks = self._service_context.prompt_helper.truncate(
                prompt=self.summary_prompt,
                text_chunks=[node.get_text() for node in half2],
            )
            text_chunk2 = "\n".join(truncated_chunks)
            summary2, _ = self._service_context.llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk2
            )
            node2 = Node(
                text=summary2,
            )
            self.index_graph.insert(node2, children_nodes=half2)

            # insert half1 and half2 as new children of parent_node
            # first remove child indices from parent node
            if parent_node is not None:
                self.index_graph.node_id_to_children_ids[
                    parent_node.get_doc_id()
                ] = list()
            else:
                self.index_graph.root_nodes = {}
            self.index_graph.insert_under_parent(
                node1, parent_node, new_index=self.index_graph.get_index(node1)
            )
            self._docstore.add_documents([node1], allow_update=False)
            self.index_graph.insert_under_parent(
                node2, parent_node, new_index=self.index_graph.get_index(node2)
            )
            self._docstore.add_documents([node2], allow_update=False)

    def _insert_node(self, node: Node, parent_node: Optional[Node] = None) -> None:
        """Insert node."""
        cur_graph_node_ids = self.index_graph.get_children(parent_node)
        cur_graph_nodes = self._docstore.get_node_dict(cur_graph_node_ids)
        cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
        # if cur_graph_nodes is empty (start with empty graph), then insert under
        # parent (insert new root node)
        if len(cur_graph_nodes) == 0:
            self._insert_under_parent_and_consolidate(node, parent_node)
        # check if leaf nodes, then just insert under parent
        elif len(self.index_graph.get_children(cur_graph_node_list[0])) == 0:
            self._insert_under_parent_and_consolidate(node, parent_node)
        # else try to find the right summary node to insert under
        else:
            text_splitter = (
                self._service_context.prompt_helper.get_text_splitter_given_prompt(
                    prompt=self.insert_prompt,
                    num_chunks=len(cur_graph_node_list),
                )
            )
            numbered_text = get_numbered_text_from_nodes(
                cur_graph_node_list, text_splitter=text_splitter
            )
            response, _ = self._service_context.llm_predictor.predict(
                self.insert_prompt,
                new_chunk_text=node.get_text(),
                num_chunks=len(cur_graph_node_list),
                context_list=numbered_text,
            )
            numbers = extract_numbers_given_response(response)
            if numbers is None or len(numbers) == 0:
                # NOTE: if we can't extract a number, then we just insert under parent
                self._insert_under_parent_and_consolidate(node, parent_node)
            elif int(numbers[0]) > len(cur_graph_node_list):
                # NOTE: if number is out of range, then we just insert under parent
                self._insert_under_parent_and_consolidate(node, parent_node)
            else:
                selected_node = cur_graph_node_list[int(numbers[0]) - 1]
                self._insert_node(node, selected_node)

        # now we need to update summary for parent node, since we
        # need to bubble updated summaries up the tree
        if parent_node is not None:
            # refetch children
            cur_graph_node_ids = self.index_graph.get_children(parent_node)
            cur_graph_nodes = self._docstore.get_node_dict(cur_graph_node_ids)
            cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
            truncated_chunks = self._service_context.prompt_helper.truncate(
                prompt=self.summary_prompt,
                text_chunks=[node.get_text() for node in cur_graph_node_list],
            )
            text_chunk = "\n".join(truncated_chunks)
            new_summary, _ = self._service_context.llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk
            )

            parent_node.text = new_summary

    def insert(self, nodes: Sequence[Node]) -> None:
        """Insert into index_graph."""
        print("calling insert")
        print(nodes)
        for node in nodes:
            self._insert_node(node)
