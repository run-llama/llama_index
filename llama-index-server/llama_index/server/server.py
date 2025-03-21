import logging
import os
from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from llama_index.core.workflow import Workflow
from llama_index.server.api.routers.chat import chat_router
from llama_index.server.chat_ui import download_chat_ui


class LlamaIndexServer(FastAPI):
    workflow_factory: Callable[..., Workflow]
    api_prefix: str = "/api"
    include_ui: Optional[bool]
    verbose: bool = False
    ui_path: str = ".ui"

    def __init__(
        self,
        workflow_factory: Callable[..., Workflow],
        logger: Optional[logging.Logger] = None,
        use_default_routers: Optional[bool] = False,
        env: Optional[str] = None,
        include_ui: Optional[bool] = None,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the LlamaIndexServer.

        Args:
            workflow_factory: A factory function that creates a workflow instance for each request.
            logger: The logger to use.
            use_default_routers: Whether to use the default routers (chat, mount `data` and `output` directories).
            env: The environment to run the server in.
            include_ui: Whether to show an chat UI in the root path.
            verbose: Whether to show verbose logs.
        """
        super().__init__(*args, **kwargs)

        self.workflow_factory = workflow_factory
        self.logger = logger or logging.getLogger("uvicorn")
        self.verbose = verbose
        self.include_ui = include_ui  # Store the explicitly passed value first

        if use_default_routers:
            self.add_default_routers()

        if str(env).lower() == "dev":
            self.allow_cors("*")
            if self.include_ui is None:
                self.include_ui = True
        if self.include_ui is None:
            self.include_ui = False

        if self.include_ui:
            self.mount_ui()

    # Default routers
    def add_default_routers(self) -> None:
        self.add_chat_router()
        self.mount_data_dir()
        self.mount_output_dir()

    def add_chat_router(self) -> None:
        """
        Add the chat router.
        """
        self.include_router(
            chat_router(
                self.workflow_factory,
                self.logger,
                self.verbose,
            ),
            prefix=self.api_prefix,
        )

    def mount_ui(self) -> None:
        """
        Mount the UI.
        """
        # Check if the static folder exists
        if self.include_ui:
            if not os.path.exists(self.ui_path):
                self.logger.warning(
                    f"UI files not found, downloading UI to {self.ui_path}"
                )
                download_chat_ui(logger=self.logger, target_path=self.ui_path)
            self._mount_static_files(directory=self.ui_path, path="/", html=True)

    def mount_data_dir(self, data_dir: str = "data") -> None:
        """
        Mount the data directory.
        """
        self._mount_static_files(
            directory=data_dir, path=f"{self.api_prefix}/files/data", html=True
        )

    def mount_output_dir(self, output_dir: str = "output") -> None:
        """
        Mount the output directory.
        """
        self._mount_static_files(
            directory=output_dir, path=f"{self.api_prefix}/files/output", html=True
        )

    def _mount_static_files(
        self, directory: str, path: str, html: bool = False
    ) -> None:
        """
        Mount static files from a directory if it exists.
        """
        if os.path.exists(directory):
            self.logger.info(f"Mounting static files '{directory}' at '{path}'")
            self.mount(
                path,
                StaticFiles(directory=directory, check_dir=False, html=html),
                name=f"{directory}-static",
            )

    def allow_cors(self, origin: str = "*") -> None:
        """
        Allow CORS for a specific origin.
        """
        self.add_middleware(
            CORSMiddleware,
            allow_origins=[origin],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
