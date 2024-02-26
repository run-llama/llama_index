"""Provides a ChatBot UI for a Github Repository. Powered by Llama Index and Panel."""

import os
import pickle
from pathlib import Path

import nest_asyncio
import panel as pn
import param
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubClient, GithubRepositoryReader

# needed because both Panel and GithubRepositoryReader starts up the ioloop
nest_asyncio.apply()

CACHE_PATH = Path(".cache/panel_chatbot")
CACHE_PATH.mkdir(parents=True, exist_ok=True)

CHAT_GPT_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/512px-ChatGPT_logo.svg.png"
CHAT_GPT_URL = "https://chat.openai.com/"
LLAMA_INDEX_LOGO = (
    "https://cdn-images-1.medium.com/max/280/1*_mrG8FG_LiD23x0-mEtUkw@2x.jpeg"
)
PANEL_LOGO = {
    "default": "https://panel.holoviz.org/_static/logo_horizontal_light_theme.png",
    "dark": "https://panel.holoviz.org/_static/logo_horizontal_dark_theme.png",
}

GITHUB_LOGO = "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"
GITHUB_URL = "https://github.com/"
LLAMA_INDEX_URL = "https://www.llamaindex.ai/"
PANEL_URL = "https://panel.holoviz.org/index.html"
GITHUB_COPILOT_LOGO = (
    "https://plugins.jetbrains.com/files/17718/447537/icon/pluginIcon.svg"
)

INDEX_NOT_LOADED = "No repository loaded"
INDEX_LOADED = "Repository loaded"
LOADING_EXISTING_DOCS = "Loading existing docs"
LOADING_NEW_DOCS = "Downloading documents"
LOADING_EXISTING_INDEX = "Loading existing index"
LOADING_NEW_INDEX = "Creating index"
CUTE_LLAMA = "https://raw.githubusercontent.com/run-llama/llama-hub/main/llama_hub/llama_packs/panel_chatbot/llama_by_sophia_yang.png"
CUTE_LLAMA_URL = "https://x.com/sophiamyang/status/1729810715467252080?s=20"

pn.chat.ChatMessage.default_avatars.update(
    {
        "assistant": GITHUB_COPILOT_LOGO,
        "user": "🦙",
    }
)
pn.chat.ChatMessage.show_reaction_icons = False

ACCENT = "#ec4899"

CSS_FIXES_TO_BE_UPSTREAMED_TO_PANEL = """
#sidebar {
    padding-left: 5px !important;
    background-color: var(--panel-surface-color);
}
.pn-wrapper {
    height: calc( 100vh - 150px);
}
.bk-active.bk-btn-primary {border-color: var(--accent-fill-active)}
.bk-btn-primary:hover {border-color: var(--accent-fill-hover)}
.bk-btn-primary {border-color: var(--accent-fill-rest)}
a {color: var(--accent-fill-rest) !important;}
a:hover {color: var(--accent-fill-hover) !important;}
"""


def _split_and_clean(cstext):
    return cstext.split(",")


class IndexLoader(pn.viewable.Viewer):
    """The IndexLoader enables the user to interactively create a VectorStoreIndex from a
    github repository of choice.
    """

    value: VectorStoreIndex = param.ClassSelector(class_=VectorStoreIndex)

    status = param.String(constant=True, doc="A status message")

    owner: str = param.String(
        default="holoviz", doc="The repository owner. For example 'holoviz'"
    )
    repo: str = param.String(
        default="panel", doc="The repository name. For example 'panel'"
    )
    filter_directories: str = param.String(
        default="examples,docs,panel",
        label="Folders",
        doc="A comma separated list of folders to include. For example 'examples,docs,panel'",
    )
    filter_file_extensions: str = param.String(
        default=".py,.md,.ipynb",
        label="File Extensions",
        doc="A comma separated list of file extensions to include. For example '.py,.md,.ipynb'",
    )

    _load = param.Event(
        label="LOAD",
        doc="Loads the repository index from the cache if it exists and otherwise from scratch",
    )
    _reload = param.Event(
        default=False,
        label="RELOAD ALL",
        doc="Loads the repository index from scratch",
    )

    def __init__(self) -> None:
        super().__init__()

        if self.index_exists:
            pn.state.execute(self.load)
        else:
            self._update_status(INDEX_NOT_LOADED)

        self._layout = pn.Column(
            self.param.owner,
            self.param.repo,
            self.param.filter_directories,
            self.param.filter_file_extensions,
            pn.pane.HTML(self.github_url),
            pn.widgets.Button.from_param(
                self.param._load,
                button_type="primary",
                disabled=self._is_loading,
                loading=self._is_loading,
            ),
            pn.widgets.Button.from_param(
                self.param._reload,
                button_type="primary",
                button_style="outline",
                disabled=self._is_loading,
                loading=self._is_loading,
            ),
            pn.pane.Markdown("### Status", margin=(3, 5)),
            pn.pane.Str(self.param.status),
        )

    def __panel__(self) -> pn.Column:
        return self._layout

    @property
    def _unique_id(self):
        uid = (
            self.owner
            + self.repo
            + self.filter_directories
            + self.filter_file_extensions
        )
        return uid.replace(",", "").replace(".", "")

    @property
    def _cached_docs_path(self):
        return CACHE_PATH / f"docs_{self._unique_id}.pickle"

    @property
    def _cached_index_path(self):
        return CACHE_PATH / f"index_{self._unique_id}.pickle"

    async def _download_docs(self):
        github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

        filter_directories = _split_and_clean(self.filter_directories)
        filter_file_extensions = _split_and_clean(self.filter_file_extensions)

        loader = GithubRepositoryReader(
            github_client,
            owner=self.owner,
            repo=self.repo,
            filter_directories=(
                filter_directories,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            filter_file_extensions=(
                filter_file_extensions,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            verbose=True,
            concurrent_requests=10,
        )
        return loader.load_data(branch="main")

    async def _get_docs(self):
        docs_path = self._cached_docs_path
        index_path = self._cached_index_path

        if docs_path.exists():
            self._update_status(LOADING_EXISTING_DOCS)
            with docs_path.open("rb") as f:
                return pickle.load(f)

        self._update_status(LOADING_NEW_DOCS)
        docs = await self._download_docs()

        with docs_path.open("wb") as f:
            pickle.dump(docs, f, pickle.HIGHEST_PROTOCOL)

        if index_path.exists():
            index_path.unlink()

        return docs

    async def _create_index(self, docs):
        return VectorStoreIndex.from_documents(docs, use_async=True)

    async def _get_index(self, index):
        index_path = self._cached_index_path

        if index_path.exists():
            self._update_status(LOADING_EXISTING_INDEX)
            with index_path.open("rb") as f:
                return pickle.load(f)

        self._update_status(LOADING_NEW_INDEX)
        index = await self._create_index(index)

        with index_path.open("wb") as f:
            pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)
        return index

    @param.depends("status")
    def _is_loading(self):
        return self.status not in [INDEX_LOADED, INDEX_NOT_LOADED]

    @param.depends("status")
    def _is_not_loading(self):
        return self.status in [INDEX_LOADED, INDEX_NOT_LOADED]

    @param.depends("_load", watch=True)
    async def load(self):
        """Loads the repository index either from the cache or by downloading from
        the repository.
        """
        self._update_status("Loading ...")
        self.value = None

        docs = await self._get_docs()
        self.value = await self._get_index(docs)
        self._update_status(INDEX_LOADED)

    @param.depends("_reload", watch=True)
    async def reload(self):
        self._update_status("Deleting cached index ...")
        if self._cached_docs_path.exists():
            self._cached_docs_path.unlink()
        if self._cached_index_path.exists():
            self._cached_index_path.unlink()

        await self.load()

    def _update_status(self, text):
        with param.edit_constant(self):
            self.status = text
        print(text)

    @param.depends("owner", "repo")
    def github_url(self):
        """Returns a html string with a link to the github repository."""
        text = f"{self.owner}/{self.repo}"
        href = f"https://github.com/{text}"
        return f"<a href='{href}' target='_blank'>{text}</a>"

    @property
    def index_exists(self):
        """Returns True if the index already exists."""
        return self._cached_docs_path.exists() and self._cached_index_path.exists()


def powered_by():
    """Returns a component describing the frameworks powering the chat ui."""
    params = {"height": 40, "sizing_mode": "fixed", "margin": (0, 10)}
    return pn.Column(
        pn.pane.Markdown("### AI Powered By", margin=(10, 5, 10, 0)),
        pn.Row(
            pn.pane.Image(LLAMA_INDEX_LOGO, link_url=LLAMA_INDEX_URL, **params),
            pn.pane.Image(CHAT_GPT_LOGO, link_url=CHAT_GPT_URL, **params),
            pn.pane.Image(PANEL_LOGO[pn.config.theme], link_url=PANEL_URL, **params),
            align="center",
        ),
    )


async def chat_component(index: VectorStoreIndex, index_loader: IndexLoader):
    """Returns the chat component powering the main area of the application."""
    if not index:
        return pn.Column(
            pn.chat.ChatMessage(
                "You are a now a *GitHub Repository assistant*.",
                user="System",
            ),
            pn.chat.ChatMessage(
                "Please **load a GitHub Repository** to start chatting with me. This can take from seconds to minutes!",
                user="Assistant",
            ),
        )

    chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

    async def generate_response(contents, user, instance):
        response = await chat_engine.astream_chat(contents)
        text = ""
        async for token in response.async_response_gen():
            text += token
            yield text

    chat_interface = pn.chat.ChatInterface(
        callback=generate_response,
        sizing_mode="stretch_both",
    )
    chat_interface.send(
        pn.chat.ChatMessage(
            "You are a now a *GitHub Repository Assistant*.", user="System"
        ),
        respond=False,
    )
    chat_interface.send(
        pn.chat.ChatMessage(
            f"Hello! you can ask me anything about {index_loader.github_url()}.",
            user="Assistant",
        ),
        respond=False,
    )
    return chat_interface


def settings_components(index_loader: IndexLoader):
    """Returns a list of the components to add to the sidebar."""
    return [
        pn.pane.Image(
            CUTE_LLAMA,
            height=250,
            align="center",
            margin=(10, 5, 25, 5),
            link_url=CUTE_LLAMA_URL,
        ),
        "## Github Repository",
        index_loader,
        powered_by(),
    ]


def create_chat_ui():
    """Returns the Chat UI."""
    pn.extension(
        sizing_mode="stretch_width", raw_css=[CSS_FIXES_TO_BE_UPSTREAMED_TO_PANEL]
    )

    index_loader = IndexLoader()

    pn.state.location.sync(
        index_loader,
        parameters={
            "owner": "owner",
            "repo": "repo",
            "filter_directories": "folders",
            "filter_file_extensions": "file_extensions",
        },
    )

    bound_chat_interface = pn.bind(
        chat_component, index=index_loader.param.value, index_loader=index_loader
    )

    return pn.template.FastListTemplate(
        title="Chat with GitHub",
        sidebar=settings_components(index_loader),
        main=[bound_chat_interface],
        accent=ACCENT,
        main_max_width="1000px",
        main_layout=None,
    )


if pn.state.served:
    create_chat_ui().servable()
