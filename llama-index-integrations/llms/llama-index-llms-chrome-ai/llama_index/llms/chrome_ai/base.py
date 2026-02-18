"""Chrome AI LLM Integration using Chrome's built-in Prompt API."""

import asyncio
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Optional, Sequence

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

logger = logging.getLogger(__name__)

# Default context window for Gemini Nano (Chrome AI)
CHROME_AI_CONTEXT_WINDOW = 6144

# JavaScript to call the Chrome Prompt API (non-streaming).
# Supports both the stable `window.LanguageModel` (Chrome 138+) and the
# earlier `window.ai.languageModel` (Chrome 127-137) surfaces.
_JS_CHAT = """
async ({ systemPrompt, userPrompt, temperature, topK }) => {
    const LM = window.LanguageModel || (window.ai && window.ai.languageModel);
    if (!LM) {
        throw new Error(
            "Chrome Prompt API not available. " +
            "Ensure you are using Chrome 127+ with the Prompt API enabled " +
            "and Gemini Nano downloaded."
        );
    }
    const params = {};
    if (systemPrompt) params.systemPrompt = systemPrompt;
    if (temperature !== null && temperature !== undefined) {
        params.temperature = temperature;
    }
    if (topK !== null && topK !== undefined) {
        params.topK = topK;
    }
    const session = await LM.create(params);
    try {
        return await session.prompt(userPrompt);
    } finally {
        session.destroy();
    }
}
"""

# JavaScript to call the Chrome Prompt API with streaming.
# Sends each incremental delta via the exposed `chromePyChunk` callback,
# then sends `null` as a sentinel when streaming is complete.
_JS_STREAM = """
async ({ systemPrompt, userPrompt, temperature, topK }) => {
    const LM = window.LanguageModel || (window.ai && window.ai.languageModel);
    if (!LM) {
        throw new Error(
            "Chrome Prompt API not available. " +
            "Ensure you are using Chrome 127+ with the Prompt API enabled " +
            "and Gemini Nano downloaded."
        );
    }
    const params = {};
    if (systemPrompt) params.systemPrompt = systemPrompt;
    if (temperature !== null && temperature !== undefined) {
        params.temperature = temperature;
    }
    if (topK !== null && topK !== undefined) {
        params.topK = topK;
    }
    const session = await LM.create(params);
    try {
        const stream = session.promptStreaming(userPrompt);
        let previousText = '';
        for await (const text of stream) {
            const delta = text.slice(previousText.length);
            if (delta) {
                await window.chromePyChunk(delta);
            }
            previousText = text;
        }
    } finally {
        session.destroy();
    }
    await window.chromePyChunk(null);
}
"""

# JavaScript to check model availability.
_JS_AVAILABILITY = """
async () => {
    const LM = window.LanguageModel || (window.ai && window.ai.languageModel);
    if (!LM) return 'unavailable';
    try {
        return await LM.availability();
    } catch (_) {
        try {
            const caps = await LM.capabilities();
            return caps.available;
        } catch (_2) {
            return 'unknown';
        }
    }
}
"""


def _extract_prompts(messages: Sequence[ChatMessage]) -> tuple:
    """Return (system_prompt, user_prompt) from a message sequence.

    The last USER message is used as the user prompt.  All SYSTEM messages are
    concatenated (in order) as the system prompt.
    """
    system_parts = []
    user_prompt = ""
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            system_parts.append(msg.content or "")
        elif msg.role == MessageRole.USER:
            user_prompt = msg.content or ""
    return "\n".join(system_parts), user_prompt


class ChromeAI(CustomLLM):
    """LLM integration for Chrome's built-in Prompt API (Gemini Nano).

    Uses `playwright-python` to drive a Chrome browser instance and interact
    with `window.LanguageModel` / `window.ai.languageModel` — Chrome's
    on-device language-model API backed by Gemini Nano.

    **Requirements**

    * Chrome 127 or later (Chrome 138+ recommended).
    * The Prompt API must be enabled in Chrome (origin-trial token or the
      ``--enable-features=PromptAPIForGeminiNano`` flag).
    * Gemini Nano must be downloaded inside Chrome
      (``chrome://components`` → *Optimization Guide On Device Model*).
    * ``playwright`` Python package installed (``pip install playwright``).
    * Playwright Chromium driver installed (``playwright install chromium``),
      though the integration launches the **real Chrome** by default so that
      the Prompt API is available.

    **Quick start**

    .. code-block:: python

        from llama_index.llms.chrome_ai import ChromeAI

        llm = ChromeAI()

        # Single-turn completion
        response = llm.complete("Explain quantum entanglement in one sentence.")
        print(response.text)

        # Chat
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="What is 2 + 2?"),
        ]
        response = llm.chat(messages)
        print(response.message.content)

        # Streaming
        for chunk in llm.stream_complete("Tell me a short story."):
            print(chunk.delta, end="", flush=True)

    .. seealso::
        `Chrome Prompt API — Chrome Platform Status
        <https://chromestatus.com/feature/5134603979063296>`_
    """

    model_name: str = Field(
        default="gemini-nano",
        description="Display name for the Chrome AI model.",
    )

    context_window: int = Field(
        default=CHROME_AI_CONTEXT_WINDOW,
        description="Maximum number of context tokens for the model.",
        gt=0,
    )

    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of output tokens to generate.",
    )

    temperature: Optional[float] = Field(
        default=None,
        description=(
            "Sampling temperature (0.0–2.0). "
            "Defaults to Chrome AI's built-in default when ``None``."
        ),
    )

    top_k: Optional[int] = Field(
        default=None,
        description=(
            "Top-k sampling parameter. "
            "Defaults to Chrome AI's built-in default when ``None``."
        ),
    )

    chrome_executable_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path to the Chrome executable. "
            "When ``None`` (default) Playwright uses the system Chrome via "
            "``channel='chrome'``."
        ),
    )

    headless: bool = Field(
        default=True,
        description="Run Chrome in headless mode.",
    )

    timeout: float = Field(
        default=60.0,
        description="Per-operation timeout in seconds.",
        gt=0,
    )

    additional_launch_args: list = Field(
        default_factory=list,
        description=(
            "Extra command-line arguments forwarded to Chrome on launch "
            "(e.g. ``['--enable-features=PromptAPIForGeminiNano']``)."
        ),
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch_options(self) -> Dict[str, Any]:
        args = list(self.additional_launch_args)
        options: Dict[str, Any] = {
            "headless": self.headless,
            "args": args,
        }
        if self.chrome_executable_path:
            options["executable_path"] = self.chrome_executable_path
        else:
            options["channel"] = "chrome"
        return options

    def _js_params(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {
            "systemPrompt": system_prompt,
            "userPrompt": user_prompt,
            "temperature": self.temperature,
            "topK": self.top_k,
        }

    # ------------------------------------------------------------------
    # Synchronous interface
    # ------------------------------------------------------------------

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from playwright.sync_api import sync_playwright

        system_prompt, user_prompt = _extract_prompts(messages)
        params = self._js_params(system_prompt, user_prompt)

        with sync_playwright() as p:
            browser = p.chromium.launch(**self._launch_options())
            page = browser.new_page()
            try:
                result: str = page.evaluate(_JS_CHAT, params)
            finally:
                browser.close()

        return ChatResponse(
            message=ChatMessage(content=result, role=MessageRole.ASSISTANT),
            raw={"text": result},
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        system_prompt, user_prompt = _extract_prompts(messages)
        params = self._js_params(system_prompt, user_prompt)

        # thread-safe queue used to pass chunks from the Playwright thread
        # (where `expose_function` callbacks are invoked) to this generator.
        chunk_queue: Queue = Queue()
        errors: list = []

        def on_chunk(delta: Optional[str]) -> None:
            chunk_queue.put(delta)

        def run_playwright() -> None:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(**self._launch_options())
                page = browser.new_page()
                page.expose_function("chromePyChunk", on_chunk)
                try:
                    page.evaluate(_JS_STREAM, params)
                except Exception as exc:
                    errors.append(exc)
                    chunk_queue.put(None)  # unblock the consumer
                finally:
                    browser.close()

        t = threading.Thread(target=run_playwright, daemon=True)
        t.start()

        text = ""
        while True:
            try:
                delta = chunk_queue.get(timeout=self.timeout)
            except Empty:
                t.join(timeout=1)
                raise TimeoutError(
                    f"Chrome AI streaming timed out after {self.timeout}s"
                )
            if delta is None:
                break
            text += delta
            yield ChatResponse(
                message=ChatMessage(content=text, role=MessageRole.ASSISTANT),
                delta=delta,
                raw={"text": text},
            )

        t.join(timeout=self.timeout)
        if errors:
            raise errors[0]

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Asynchronous interface
    # ------------------------------------------------------------------

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        from playwright.async_api import async_playwright

        system_prompt, user_prompt = _extract_prompts(messages)
        params = self._js_params(system_prompt, user_prompt)

        async with async_playwright() as p:
            browser = await p.chromium.launch(**self._launch_options())
            page = await browser.new_page()
            try:
                result: str = await page.evaluate(_JS_CHAT, params)
            finally:
                await browser.close()

        return ChatResponse(
            message=ChatMessage(content=result, role=MessageRole.ASSISTANT),
            raw={"text": result},
        )

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        from playwright.async_api import async_playwright

        system_prompt, user_prompt = _extract_prompts(messages)
        params = self._js_params(system_prompt, user_prompt)

        queue: asyncio.Queue = asyncio.Queue()

        # expose_function supports async callbacks in the async Playwright API.
        def on_chunk(delta: Optional[str]) -> None:
            queue.put_nowait(delta)

        async def run_streaming() -> None:
            async with async_playwright() as p:
                browser = await p.chromium.launch(**self._launch_options())
                page = await browser.new_page()
                await page.expose_function("chromePyChunk", on_chunk)
                try:
                    await page.evaluate(_JS_STREAM, params)
                finally:
                    await browser.close()

        task = asyncio.create_task(run_streaming())

        text = ""
        try:
            while True:
                delta = await asyncio.wait_for(
                    queue.get(), timeout=self.timeout
                )
                if delta is None:
                    break
                text += delta
                yield ChatResponse(
                    message=ChatMessage(content=text, role=MessageRole.ASSISTANT),
                    delta=delta,
                    raw={"text": text},
                )
        except asyncio.TimeoutError:
            task.cancel()
            raise TimeoutError(
                f"Chrome AI async streaming timed out after {self.timeout}s"
            )
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def check_availability(self) -> str:
        """Return Chrome AI availability status.

        Returns one of: ``'available'``, ``'downloadable'``,
        ``'downloading'``, ``'unavailable'``, or ``'unknown'``.

        .. code-block:: python

            llm = ChromeAI()
            print(llm.check_availability())  # e.g. "available"
        """
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(**self._launch_options())
            page = browser.new_page()
            try:
                result: str = page.evaluate(_JS_AVAILABILITY)
            finally:
                browser.close()

        return result
