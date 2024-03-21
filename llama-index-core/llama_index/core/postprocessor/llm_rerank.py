from typing import Callable, List, Optional

class LLMRerank:
    """LLM-based reranker.

    This class represents an LLM-based reranker, which reorders a list of nodes based on their relevance to a given query using a Language Model (LLM).

    Attributes:
        top_n (int): The number of top-ranked nodes to return.
        choice_select_prompt (BasePromptTemplate): The prompt template used for choice selection.
        choice_batch_size (int): The batch size for choice selection.
        llm (LLM): The Language Model used for reranking.
    """

    top_n: int
    choice_select_prompt: "BasePromptTemplate"
    choice_batch_size: int
    llm: "LLM"

    def __init__(
        self,
        llm: Optional["LLM"] = None,
        choice_select_prompt: Optional["BasePromptTemplate"] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional["ServiceContext"] = None,
        top_n: int = 10,
    ) -> None:
        """Initialize LLMRerank.

        Args:
            llm (Optional[LLM]): The LLM instance to use. Defaults to None.
            choice_select_prompt (Optional[BasePromptTemplate]): The prompt template for choice selection. Defaults to None.
            choice_batch_size (int): The batch size for choice selection. Defaults to 10.
            format_node_batch_fn (Optional[Callable]): Function to format node batch. Defaults to None.
            parse_choice_select_answer_fn (Optional[Callable]): Function to parse choice select answer. Defaults to None.
            service_context (Optional[ServiceContext]): Service context. Defaults to None.
            top_n (int): The number of top-ranked nodes to return. Defaults to 10.
        """
        choice_select_prompt = choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT

        llm = llm or llm_from_settings_or_context(Settings, service_context)

        self.choice_select_prompt = choice_select_prompt
        self.choice_batch_size = choice_batch_size
        self.llm = llm
        self.top_n = top_n
        self.format_node_batch_fn = format_node_batch_fn or default_format_node_batch_fn
        self.parse_choice_select_answer_fn = parse_choice_select_answer_fn or default_parse_choice_select_answer_fn

    def _get_prompts(self) -> "PromptDictType":
        """Get prompts."""
        return {"choice_select_prompt": self.choice_select_prompt}

    def _update_prompts(self, prompts: "PromptDictType") -> None:
        """Update prompts."""
        if "choice_select_prompt" in prompts:
            self.choice_select_prompt = prompts["choice_select_prompt"]

    def _postprocess_nodes(
        self,
        nodes: List["NodeWithScore"],
        query_bundle: Optional["QueryBundle"] = None,
    ) -> List["NodeWithScore"]:
        """Rerank nodes based on LLM predictions.

        Args:
            nodes (List[NodeWithScore]): The list of nodes to rerank.
            query_bundle (Optional[QueryBundle]): The query bundle. Defaults to None.

        Returns:
            List[NodeWithScore]: The reranked nodes.

        Raises:
            ValueError: If query bundle is not provided.

        """
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        initial_results: List["NodeWithScore"] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [
                node.node for node in nodes[idx : idx + self.choice_batch_size]
            ]

            query_str = query_bundle.query_str
            fmt_batch_str = self.format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response = self.llm.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self.parse_choice_select_answer_fn(
                raw_response, len(nodes_batch)
            )
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
            relevances = relevances or [1.0 for _ in choice_nodes]
            initial_results.extend(
                [
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]

    def compare_with_cohere_rerank(self) -> str:
        """Compare with Cohere Rerank.

        This method provides a comparison between LLMRerank and Cohere Rerank in terms of performance and pricing.

        Returns:
            str: Comparison between LLMRerank and Cohere Rerank.

        """
        comparison_details = """
        LLMRerank vs. Cohere Rerank:

        Performance:
        - LLMRerank leverages Language Models (LLMs) for reranking, offering flexibility in model selection and customization.
        - Cohere Rerank utilizes the capabilities of the Cohere API for reranking.

        Pricing:
        - Pricing for LLMRerank may vary based on the selected Language Model provider and usage.
        - Cohere Rerank's pricing is determined by the pricing plans of the Cohere API.

        Conclusion:
        - LLMRerank offers flexibility and potentially lower costs depending on the selected Language Model provider.
        - Cohere Rerank provides a convenient solution integrated with the Cohere API but may have specific pricing considerations.
        """

        return comparison_details.strip()
