"""iGPT Email Intelligence tool spec."""

import json
from typing import List, Optional

from igptai import IGPT

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class IGPTEmailToolSpec(BaseToolSpec):
    """
    iGPT Email Intelligence tool spec.

    Wraps the iGPT recall.ask() and recall.search() endpoints, giving agents
    structured, reasoning-ready context from connected email threads.

    Args:
        api_key (str): iGPT API key. See https://docs.igpt.ai for details.
        user (str): User identifier for the connected mailbox.

    Example:
        .. code-block:: python

            from llama_index.tools.igpt_email import IGPTEmailToolSpec
            from llama_index.core.agent.workflow import FunctionAgent
            from llama_index.llms.openai import OpenAI

            tool_spec = IGPTEmailToolSpec(api_key="your-key", user="user-id")

            agent = FunctionAgent(
                tools=tool_spec.to_tool_list(),
                llm=OpenAI(model="gpt-4.1"),
            )

            answer = await agent.run("What tasks were assigned to me this week?")

    """

    spec_functions = ["ask", "search"]

    def __init__(self, api_key: str, user: str) -> None:
        """Initialize with parameters."""
        self.client = IGPT(api_key=api_key, user=user)

    def ask(
        self,
        question: str,
        output_format: str = "json",
    ) -> List[Document]:
        """
        Ask a question about email context using iGPT's reasoning engine.

        Calls recall.ask() and returns structured context extracted from
        connected email threads, including tasks, decisions, owners, sentiment,
        deadlines, and citations.

        Args:
            question (str): The question or prompt to reason over email context.
            output_format (str): Response format — "text" or "json". Default is "json".

        Returns:
            List[Document]: A single Document containing the structured reasoning
                response. Citations are stored in metadata["citations"].

        """
        response = self.client.recall.ask(
            input=question,
            output_format=output_format,
        )

        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"iGPT API error: {response['error']}")

        if isinstance(response, dict):
            text = json.dumps(response)
            citations = response.get("citations", [])
        else:
            text = str(response)
            citations = []

        return [
            Document(
                text=text,
                metadata={
                    "question": question,
                    "citations": citations,
                    "source": "igpt_email_ask",
                },
            )
        ]

    def search(
        self,
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """
        Search email context for relevant messages and threads.

        Calls recall.search() and returns matching email context as Documents,
        with thread metadata (subject, participants, date, thread ID) preserved
        in metadata for downstream filtering and retrieval.

        Args:
            query (str): Search query to run against connected email data.
            date_from (str, optional): Filter results from this date (YYYY-MM-DD).
            date_to (str, optional): Filter results up to this date (YYYY-MM-DD).
            max_results (int, optional): Maximum number of results to return. Default is 10.

        Returns:
            List[Document]: One Document per email result. Thread metadata is
                stored in metadata (subject, from, to, date, thread_id, id).

        """
        response = self.client.recall.search(
            query=query,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
        )

        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"iGPT API error: {response['error']}")

        if not response:
            return []

        results = (
            response if isinstance(response, list) else response.get("results", [])
        )

        documents = []
        for item in results:
            if isinstance(item, dict):
                text = item.get("content", item.get("body", json.dumps(item)))
                metadata = {
                    "source": "igpt_email_search",
                    "subject": item.get("subject"),
                    "from": item.get("from"),
                    "to": item.get("to"),
                    "date": item.get("date"),
                    "thread_id": item.get("thread_id"),
                    "id": item.get("id"),
                }
            else:
                text = str(item)
                metadata = {"source": "igpt_email_search"}

            documents.append(Document(text=text, metadata=metadata))

        return documents
