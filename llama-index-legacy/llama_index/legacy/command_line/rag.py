import asyncio
import os
import shutil
from argparse import ArgumentParser
from glob import iglob
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, cast

from llama_index.legacy import (
    Response,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.legacy.bridge.pydantic import BaseModel, Field, validator
from llama_index.legacy.chat_engine import CondenseQuestionChatEngine
from llama_index.legacy.core.response.schema import RESPONSE_TYPE, StreamingResponse
from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.legacy.ingestion import IngestionPipeline
from llama_index.legacy.llms import LLM, OpenAI
from llama_index.legacy.query_engine import CustomQueryEngine
from llama_index.legacy.query_pipeline import FnComponent
from llama_index.legacy.query_pipeline.query import QueryPipeline
from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.response_synthesizers import CompactAndRefine
from llama_index.legacy.utils import get_cache_dir

RAG_HISTORY_FILE_NAME = "files_history.txt"


def default_ragcli_persist_dir() -> str:
    return str(Path(get_cache_dir()) / "rag_cli")


def query_input(query_str: Optional[str] = None) -> str:
    return query_str or ""


class QueryPipelineQueryEngine(CustomQueryEngine):
    query_pipeline: QueryPipeline = Field(
        description="Query Pipeline to use for Q&A.",
    )

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        return self.query_pipeline.run(query_str=query_str)

    async def acustom_query(self, query_str: str) -> RESPONSE_TYPE:
        return await self.query_pipeline.arun(query_str=query_str)


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
    llm: LLM = Field(
        description="Language model to use for response generation.",
        default_factory=lambda: OpenAI(model="gpt-3.5-turbo", streaming=True),
    )
    query_pipeline: Optional[QueryPipeline] = Field(
        description="Query Pipeline to use for Q&A.",
        default=None,
    )
    chat_engine: Optional[CondenseQuestionChatEngine] = Field(
        description="Chat engine to use for chatting.",
        default_factory=None,
    )
    file_extractor: Optional[Dict[str, BaseReader]] = Field(
        description="File extractor to use for extracting text from files.",
        default=None,
    )

    class Config:
        arbitrary_types_allowed = True

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
        query_component = FnComponent(
            fn=query_input, output_key="output", req_params={"query_str"}
        )
        llm = cast(LLM, values["llm"])

        # get embed_model from transformations if possible
        embed_model = None
        if ingestion_pipeline.transformations is not None:
            for transformation in ingestion_pipeline.transformations:
                if isinstance(transformation, BaseEmbedding):
                    embed_model = transformation
                    break

        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model or "default"
        )
        retriever = VectorStoreIndex.from_vector_store(
            ingestion_pipeline.vector_store, service_context=service_context
        ).as_retriever(similarity_top_k=8)
        response_synthesizer = CompactAndRefine(
            service_context=service_context, streaming=True, verbose=verbose
        )

        # define query pipeline
        query_pipeline = QueryPipeline(verbose=verbose)
        query_pipeline.add_modules(
            {
                "query": query_component,
                "retriever": retriever,
                "summarizer": response_synthesizer,
            }
        )
        query_pipeline.add_link("query", "retriever")
        query_pipeline.add_link("retriever", "summarizer", dest_key="nodes")
        query_pipeline.add_link("query", "summarizer", dest_key="query_str")
        return query_pipeline

    @validator("chat_engine", always=True)
    def chat_engine_from_query_pipeline(
        cls, chat_engine: Any, values: Dict[str, Any]
    ) -> Optional[CondenseQuestionChatEngine]:
        """
        If chat_engine is not provided, create one from query_pipeline.
        """
        if chat_engine is not None:
            return chat_engine

        if values.get("query_pipeline", None) is None:
            values["query_pipeline"] = cls.query_pipeline_from_ingestion_pipeline(
                query_pipeline=None, values=values
            )

        query_pipeline = cast(QueryPipeline, values["query_pipeline"])
        if query_pipeline is None:
            return None
        query_engine = QueryPipelineQueryEngine(query_pipeline=query_pipeline)  # type: ignore
        verbose = cast(bool, values["verbose"])
        llm = cast(LLM, values["llm"])
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine, llm=llm, verbose=verbose
        )

    async def handle_cli(
        self,
        files: Optional[str] = None,
        question: Optional[str] = None,
        chat: bool = False,
        verbose: bool = False,
        clear: bool = False,
        create_llama: bool = False,
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
                    reader = SimpleDirectoryReader(
                        input_dir=_file,
                        filename_as_id=True,
                        file_extractor=self.file_extractor,
                    )
                else:
                    reader = SimpleDirectoryReader(
                        input_files=[_file],
                        filename_as_id=True,
                        file_extractor=self.file_extractor,
                    )

                documents.extend(reader.load_data(show_progress=verbose))

            await ingestion_pipeline.arun(show_progress=verbose, documents=documents)
            ingestion_pipeline.persist(persist_dir=self.persist_dir)

            # Append the `--files` argument to the history file
            with open(f"{self.persist_dir}/{RAG_HISTORY_FILE_NAME}", "a") as f:
                f.write(files + "\n")

        if create_llama:
            if shutil.which("npx") is None:
                print(
                    "`npx` is not installed. Please install it by calling `npm install -g npx`"
                )
            else:
                history_file_path = Path(f"{self.persist_dir}/{RAG_HISTORY_FILE_NAME}")
                if not history_file_path.exists():
                    print(
                        "No data has been ingested, "
                        "please specify `--files` to create llama dataset."
                    )
                else:
                    with open(history_file_path) as f:
                        stored_paths = {line.strip() for line in f if line.strip()}
                    if len(stored_paths) == 0:
                        print(
                            "No data has been ingested, "
                            "please specify `--files` to create llama dataset."
                        )
                    elif len(stored_paths) > 1:
                        print(
                            "Multiple files or folders were ingested, which is not supported by create-llama. "
                            "Please call `llamaindex-cli rag --clear` to clear the cache first, "
                            "then call `llamaindex-cli rag --files` again with a single folder or file"
                        )
                    else:
                        path = stored_paths.pop()
                        if "*" in path:
                            print(
                                "Glob pattern is not supported by create-llama. "
                                "Please call `llamaindex-cli rag --clear` to clear the cache first, "
                                "then call `llamaindex-cli rag --files` again with a single folder or file."
                            )
                        elif not os.path.exists(path):
                            print(
                                f"The path {path} does not exist. "
                                "Please call `llamaindex-cli rag --clear` to clear the cache first, "
                                "then call `llamaindex-cli rag --files` again with a single folder or file."
                            )
                        else:
                            print(f"Calling create-llama using data from {path} ...")
                            command_args = [
                                "npx",
                                "create-llama@latest",
                                "--frontend",
                                "--template",
                                "streaming",
                                "--framework",
                                "fastapi",
                                "--ui",
                                "shadcn",
                                "--vector-db",
                                "none",
                                "--engine",
                                "context",
                                f"--files {path}",
                            ]
                            os.system(" ".join(command_args))

        if question is not None:
            await self.handle_question(question)
        if chat:
            await self.start_chat_repl()

    async def handle_question(self, question: str) -> None:
        if self.query_pipeline is None:
            raise ValueError("query_pipeline is not defined.")
        query_pipeline = cast(QueryPipeline, self.query_pipeline)
        query_pipeline.verbose = self.verbose
        chat_engine = cast(CondenseQuestionChatEngine, self.chat_engine)
        response = chat_engine.chat(question)

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
        chat_engine = cast(CondenseQuestionChatEngine, self.chat_engine)
        chat_engine.streaming_chat_repl()

    @classmethod
    def add_parser_args(
        cls,
        parser: Union[ArgumentParser, Any],
        instance_generator: Callable[[], "RagCLI"],
    ) -> None:
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
                'such as "file.pdf".'
            ),
        )
        parser.add_argument(
            "-c",
            "--chat",
            help="If flag is present, opens a chat REPL.",
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
        parser.add_argument(
            "--create-llama",
            help="Create a LlamaIndex application with your embedded data.",
            required=False,
            action="store_true",
        )
        parser.set_defaults(
            func=lambda args: asyncio.run(instance_generator().handle_cli(**vars(args)))
        )

    def cli(self) -> None:
        """
        Entrypoint for CLI tool.
        """
        parser = ArgumentParser(description="LlamaIndex RAG Q&A tool.")
        subparsers = parser.add_subparsers(
            title="commands", dest="command", required=True
        )
        llamarag_parser = subparsers.add_parser(
            "rag", help="Ask a question to a document / a directory of documents."
        )
        self.add_parser_args(llamarag_parser, lambda: self)
        # Parse the command-line arguments
        args = parser.parse_args()

        # Call the appropriate function based on the command
        args.func(args)
