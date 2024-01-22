import asyncio
import os
from argparse import ArgumentParser
from glob import iglob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from llama_index import (
    PromptTemplate,
    Response,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.bridge.pydantic import BaseModel, Field, validator
from llama_index.core.response.schema import StreamingResponse
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import OpenAI
from llama_index.query_pipeline.query import QueryPipeline
from llama_index.response_synthesizers import CompactAndRefine
from llama_index.schema import Document
from llama_index.utils import get_cache_dir


def default_ragcli_persist_dir() -> str:
    return str(Path(get_cache_dir()) / "rag_cli")


class RagCLI(BaseModel):
    """
    CLI tool for chatting with output of a IngestionPipeline via a QueryPipeline.
    """

    ingestion_pipeline: IngestionPipeline = Field(
        description="Ingestion pipeline to run for RAG ingestion."
    )
    verbose: bool = Field(
        description="Whether to print out verbose information during execution.",
        default=False,
    )
    persist_dir: str = Field(
        description="Directory to persist ingestion pipeline.",
        default_factory=default_ragcli_persist_dir,
    )
    query_pipeline: Optional[QueryPipeline] = Field(
        description="Query Pipeline to use for Q&A.",
        default=None,
    )
    on_ingestion: Callable[[List[Document]], None] = Field(
        description="Callback to run after ingestion.",
        default=lambda documents: None,
    )

    @validator("query_pipeline", always=True)
    def query_pipeline_from_ingestion_pipeline(
        cls, query_pipeline: Any, values: Dict[str, Any]
    ) -> Optional[QueryPipeline]:
        """
        If query_pipeline is not provided, create one from ingestion_pipeline.
        """
        if query_pipeline is not None:
            return query_pipeline

        ingestion_pipeline = cast(IngestionPipeline, values["ingestion_pipeline"])
        if ingestion_pipeline.vector_store is None:
            return None
        verbose = cast(bool, values["verbose"])
        prompt_tmpl = PromptTemplate("{question}")
        retriever = VectorStoreIndex.from_vector_store(
            ingestion_pipeline.vector_store
        ).as_retriever(similarity_top_k=5)
        response_synthesizer = CompactAndRefine(streaming=True)
        llm = OpenAI(model="gpt-3.5-turbo", streaming=True)
        # define query pipeline
        query_pipeline = QueryPipeline(verbose=verbose)
        query_pipeline.add_modules(
            {
                "llm": llm,
                "prompt_tmpl": prompt_tmpl,
                "retriever": retriever,
                "summarizer": response_synthesizer,
            }
        )
        query_pipeline.add_link("prompt_tmpl", "llm")
        query_pipeline.add_link("llm", "retriever")
        query_pipeline.add_link("retriever", "summarizer", dest_key="nodes")
        query_pipeline.add_link("prompt_tmpl", "summarizer", dest_key="query_str")
        return query_pipeline

    async def handle_cli(
        self,
        files: Optional[str] = None,
        question: Optional[str] = None,
        chat: bool = False,
        verbose: bool = False,
        clear: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Entrypoint for local document RAG CLI tool.
        """
        if clear:
            # delete self.persist_dir directory including all subdirectories and files
            if os.path.exists(self.persist_dir):
                # Ask for confirmation
                response = input(
                    f"Are you sure you want to delete data within {self.persist_dir}? [y/N] "
                )
                if response.strip().lower() != "y":
                    print("Aborted.")
                    return
                os.system(f"rm -rf {self.persist_dir}")
            print(f"Successfully cleared {self.persist_dir}")

        self.verbose = verbose
        ingestion_pipeline = cast(IngestionPipeline, self.ingestion_pipeline)
        if self.verbose:
            print("Saving/Loading from persist_dir: ", self.persist_dir)
        if files is not None:
            documents = []
            for _file in iglob(files, recursive=True):
                _file = os.path.abspath(_file)
                if os.path.isdir(_file):
                    reader = SimpleDirectoryReader(input_dir=_file, filename_as_id=True)
                else:
                    reader = SimpleDirectoryReader(
                        input_files=[_file], filename_as_id=True
                    )

                documents.extend(reader.load_data(show_progress=verbose))

            await ingestion_pipeline.arun(show_progress=verbose, documents=documents)
            ingestion_pipeline.persist(persist_dir=self.persist_dir)
            self.on_ingestion(documents)

        if question is not None:
            await self.handle_question(question)
        if chat:
            await self.start_chat_repl()

    async def handle_question(self, question: str) -> None:
        if self.query_pipeline is None:
            raise ValueError("query_pipeline is not defined.")
        query_pipeline = cast(QueryPipeline, self.query_pipeline)
        query_pipeline.verbose = self.verbose
        response = query_pipeline.run(question=question)

        if isinstance(response, StreamingResponse):
            response.print_response_stream()
        else:
            response = cast(Response, response)
            print(response)

    async def start_chat_repl(self) -> None:
        """
        Start a REPL for chatting with the agent.
        """
        if self.query_pipeline is None:
            raise ValueError("query_pipeline is not defined.")
        while True:
            question = input("\n(rag) ")
            await self.handle_question(question.strip())

    def add_parser_args(self, parser: Union[ArgumentParser, Any]) -> None:
        parser.add_argument(
            "-q",
            "--question",
            type=str,
            help="The question you want to ask.",
            required=False,
        )

        parser.add_argument(
            "-f",
            "--files",
            type=str,
            help=(
                "The name of the file or directory you want to ask a question about,"
                "such as `file.pdf`."
            ),
        )
        parser.add_argument(
            "-c",
            "--chat",
            help="Whether to open a chat REPL.",
            action="store_true",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            help="Whether to print out verbose information during execution.",
            action="store_true",
        )
        parser.add_argument(
            "--clear",
            help="Clears out all currently embedded data.",
            action="store_true",
        )
        parser.set_defaults(
            func=lambda args: asyncio.run(self.handle_cli(**vars(args)))
        )

    def cli(self) -> None:
        """
        Entrypoint for CLI tool.
        """
        parser = ArgumentParser(description="LlamaIndex RAG Q&A tool.")
        subparsers = parser.add_subparsers(
            title="commands", dest="command", required=True
        )
        self.add_parser_args(subparsers)
        # Parse the command-line arguments
        args = parser.parse_args()

        # Call the appropriate function based on the command
        args.func(args)
