"""Dashscope Agent for Alibaba cloud bailian."""

from http import HTTPStatus
from typing import (
    List,
    Optional,
    Union,
)

from dashscope import Application
from dashscope.app.application_response import ApplicationResponse
from llama_index.core.agent.types import BaseAgent
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
    AgentChatResponse,
)


class DashScopeAgent(BaseAgent):
    """
    DashScope agent simple wrapper for Alibaba cloud bailian high-level agent api.
    """

    def __init__(
        self,
        app_id: str,
        chat_session: bool = True,
        workspace: str = None,
        api_key: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Init params.

        Args:
            app_id (str): id of Alibaba cloud bailian application
            chat_session (bool): When need to keep chat session, defaults to True.
            workspace(str, `optional`): Workspace of Alibaba cloud bailian
            api_key (str, optional): The api api_key, can be None,
                if None, will get from ENV DASHSCOPE_API_KEY.
            verbose: Output verbose info or not.

        """
        self.app_id = app_id
        self.chat_session = chat_session
        self.workspace = workspace
        self.api_key = api_key
        self._verbose = verbose
        self._session_id = None

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None, **kwargs
    ) -> AgentChatResponse:
        return self._chat(message=message, stream=False, **kwargs)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        raise NotImplementedError("achat not implemented")

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None, **kwargs
    ) -> StreamingAgentChatResponse:
        return self._chat(message=message, stream=True, **kwargs)

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")

    def reset(self) -> None:
        self._session_id = None

    @property
    def chat_history(self) -> List[ChatMessage]:
        raise NotImplementedError("chat_history not implemented")

    @property
    def get_session_id(self) -> str:
        return self._session_id

    def _chat(
        self,
        message: str,
        stream: bool = False,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs,
    ) -> Union[AgentChatResponse, StreamingAgentChatResponse]:
        """
        Call app completion service.

        Args:
            message (str): Message for chatting with LLM.
            chat_history (List[ChatMessage], `optional`): The user provided chat history. Defaults to None.

            **kwargs:
                session_id(str, `optional`): Session if for multiple rounds call.
                biz_params(dict, `optional`): The extra parameters for flow or plugin.

        Raises:
            ValueError: The request failed with http code and message.

        Returns:
            Union[AgentChatResponse, StreamingAgentChatResponse]

        """
        if stream:
            kwargs["stream"] = True

        if self.chat_session:
            kwargs["session_id"] = self._session_id

        response = Application.call(
            app_id=self.app_id,
            prompt=message,
            history=None,
            workspace=self.workspace,
            api_key=self.api_key,
            **kwargs,
        )

        if stream:
            return StreamingAgentChatResponse(
                chat_stream=(self.from_dashscope_response(rsp) for rsp in response)
            )
        else:
            if response.status_code != HTTPStatus.OK:
                raise ValueError(
                    f"Chat failed with status: {response.status_code}, request id: {response.request_id}, "
                    f"code: {response.code}, message: {response.message}"
                )

            if self._verbose:
                print("Got chat response: %s" % response)
            self._session_id = response.output.session_id

            return AgentChatResponse(response=response.output.text)

    def from_dashscope_response(self, response: ApplicationResponse) -> ChatResponse:
        if response.status_code != HTTPStatus.OK:
            raise ValueError(
                f"Chat failed with status: {response.status_code}, request id: {response.request_id}, "
                f"code: {response.code}, message: {response.message}"
            )

        if self._verbose and response.output.finish_reason == "stop":
            print("Got final chat response: %s" % response)
        self._session_id = response.output.session_id

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.output.text
            )
        )
