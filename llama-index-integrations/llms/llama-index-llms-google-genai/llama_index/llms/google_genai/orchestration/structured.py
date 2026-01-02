from __future__ import annotations

import asyncio

from typing import (
    Any,
    AsyncGenerator,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import google.genai
import google.genai.types as types
from llama_index.core.llms.llm import Model
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import BaseModel, ValidationError

from llama_index.core.program.utils import FlexibleModel, create_flexible_model
from llama_index.core.program.utils import _repair_incomplete_json  # noqa

from llama_index.llms.google_genai.conversion.messages import MessageConverter
from llama_index.llms.google_genai.files import FileManager


class StructuredStreamParser:
    def __init__(self, *, output_cls: Type[Model]) -> None:
        self._output_cls = output_cls
        self._flexible_model: Type[FlexibleModel] = create_flexible_model(output_cls)  # type: ignore[arg-type]
        self._current_json = ""

    def update(self, chunk: types.GenerateContentResponse) -> Optional[Model]:
        if isinstance(chunk.parsed, BaseModel):
            return chunk.parsed  # type: ignore[return-value]
        if chunk.parsed is not None:
            return self._output_cls.model_validate_json(str(chunk.parsed))

        data = self._chunk_text_delta(chunk)
        if not data:
            return None

        self._current_json += data
        try:
            return self._output_cls.model_validate_json(self._current_json)
        except ValidationError:
            try:
                return self._flexible_model.model_validate_json(
                    _repair_incomplete_json(self._current_json)
                )
            except ValidationError:
                return None

    @staticmethod
    def _chunk_text_delta(chunk: types.GenerateContentResponse) -> Optional[str]:
        if not chunk.candidates:
            return None
        candidate = chunk.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return None

        # Structured output streams JSON text; be robust and join all text parts.
        texts = [p.text for p in candidate.content.parts if p.text]
        return "".join(texts) if texts else None


class PreparedStructured:
    def __init__(
        self,
        *,
        model: str,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
        output_cls: Type[Model],
        uploaded_file_names: List[str],
    ) -> None:
        self.model = model
        self.contents = contents
        self.config = config
        self.output_cls = output_cls
        self.uploaded_file_names = uploaded_file_names


class StructuredRunner:
    """Run structured prediction requests using the Gemini models API."""

    _JSON_MIME_TYPE = "application/json"

    def __init__(
        self,
        *,
        client: google.genai.Client,
        model: str,
        file_manager: FileManager,
        message_converter: MessageConverter,
    ) -> None:
        self._client = client
        self._model = model
        self._file_manager = file_manager
        self._message_converter = message_converter

    async def prepare(
        self,
        *,
        messages: Sequence[ChatMessage],
        output_cls: Type[Model],
        **kwargs: Any,
    ) -> PreparedStructured:
        generation_config = kwargs.pop("generation_config", None)

        contents, uploaded_file_names = await self._prepare_contents(messages)
        prepared_config = self._prepare_config(
            output_cls=output_cls,
            generation_config=generation_config,
        )

        return PreparedStructured(
            model=self._model,
            contents=contents,
            config=prepared_config,
            output_cls=output_cls,
            uploaded_file_names=uploaded_file_names,
        )

    def _run(self, prepared: PreparedStructured):
        try:
            response = self._client.models.generate_content(
                model=prepared.model,
                contents=prepared.contents,
                config=prepared.config,
            )
        finally:
            if self._file_manager.file_mode in ("fileapi", "hybrid"):
                self._file_manager.cleanup(prepared.uploaded_file_names)

        return response

    def run(
        self,
        prepared: PreparedStructured,
    ) -> Model:
        response = self._run(prepared)
        if isinstance(response.parsed, BaseModel):
            return response.parsed  # type: ignore[return-value]
        return prepared.output_cls.model_validate_json(response.text)

    def run_parsed(self, prepared: PreparedStructured) -> Model:
        response = self._run(prepared)
        if isinstance(response.parsed, BaseModel):
            return response.parsed  # type: ignore[return-value]
        raise ValueError("Response is not a BaseModel")

    async def arun(
        self,
        prepared: PreparedStructured,
    ) -> Model:
        try:
            response = await self._client.aio.models.generate_content(
                model=prepared.model,
                contents=prepared.contents,
                config=prepared.config,
            )
        finally:
            if self._file_manager.file_mode in ("fileapi", "hybrid"):
                await self._file_manager.acleanup(prepared.uploaded_file_names)

        if isinstance(response.parsed, BaseModel):
            return response.parsed  # type: ignore[return-value]

        return prepared.output_cls.model_validate_json(response.text)

    def stream(
        self,
        prepared: PreparedStructured,
    ) -> Generator[Union[Model, Any], None, None]:
        def gen() -> Generator[Union[Model, Any], None, None]:
            parser = StructuredStreamParser(output_cls=prepared.output_cls)
            try:
                response_gen = self._client.models.generate_content_stream(
                    model=prepared.model,
                    contents=prepared.contents,
                    config=prepared.config,
                )

                for chunk in response_gen:
                    parsed = parser.update(chunk)
                    if parsed is not None:
                        yield parsed
            finally:
                if self._file_manager.file_mode in ("fileapi", "hybrid"):
                    self._file_manager.cleanup(prepared.uploaded_file_names)

        return gen()

    async def astream(
        self,
        prepared: PreparedStructured,
    ) -> AsyncGenerator[Union[Model, Any], None]:
        async def gen() -> AsyncGenerator[Union[Model, Any], None]:
            parser = StructuredStreamParser(output_cls=prepared.output_cls)
            try:
                response_stream = await self._client.aio.models.generate_content_stream(
                    model=prepared.model,
                    contents=prepared.contents,
                    config=prepared.config,
                )
                async for chunk in response_stream:
                    parsed = parser.update(chunk)
                    if parsed is not None:
                        yield parsed
            finally:
                if self._file_manager.file_mode in ("fileapi", "hybrid"):
                    await self._file_manager.acleanup(prepared.uploaded_file_names)

        return gen()

    def _prepare_config(
        self,
        *,
        output_cls: Type[Model],
        generation_config: Optional[dict],
    ) -> types.GenerateContentConfig:
        config_dict: dict = dict(generation_config or {})

        config_dict["response_mime_type"] = self._JSON_MIME_TYPE
        config_dict["response_schema"] = output_cls

        return types.GenerateContentConfig(**config_dict)

    async def _prepare_contents(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[List[types.Content], List[str]]:
        contents_and_names = await asyncio.gather(
            *(self._message_converter.to_gemini_content(m) for m in list(messages))
        )
        contents = [it[0] for it in contents_and_names if it[0] is not None]
        uploaded_file_names = [name for it in contents_and_names for name in it[1]]
        return contents, uploaded_file_names
