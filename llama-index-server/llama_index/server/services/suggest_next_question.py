import logging
import os
import re
from typing import List, Optional

from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.server.api.models import ChatAPIMessage

logger = logging.getLogger("uvicorn")


class SuggestNextQuestionsService:
    """
    Suggest the next questions that user might ask based on the conversation history.
    """

    prompt = PromptTemplate(
        r"""
You're a helpful assistant! Your task is to suggest the next questions that user might interested in to keep the conversation going.
Here is the conversation history
---------------------
{conversation}
---------------------
Given the conversation history, please give me 3 questions that user might ask next!
Your answer should be wrapped in three sticks without any index numbers and follows the following format:
\`\`\`
<question 1>
<question 2>
<question 3>
\`\`\`
"""
    )

    @classmethod
    def get_configured_prompt(cls) -> PromptTemplate:
        prompt = os.getenv("NEXT_QUESTION_PROMPT", None)
        if not prompt:
            return cls.prompt
        return PromptTemplate(prompt)

    @classmethod
    async def suggest_next_questions_all_messages(
        cls,
        messages: List[ChatAPIMessage],
    ) -> Optional[List[str]]:
        """
        Suggest the next questions that user might ask based on the conversation history.
        """
        prompt_template = cls.get_configured_prompt()

        try:
            # Reduce the cost by only using the last two messages
            last_user_message = None
            last_assistant_message = None
            for message in reversed(messages):
                if message.role == "user":
                    last_user_message = f"User: {message.content}"
                elif message.role == "assistant":
                    last_assistant_message = f"Assistant: {message.content}"
                if last_user_message and last_assistant_message:
                    break
            conversation: str = f"{last_user_message}\n{last_assistant_message}"

            # Call the LLM and parse questions from the output
            prompt = prompt_template.format(conversation=conversation)
            output = await Settings.llm.acomplete(prompt)
            return cls._extract_questions(output.text)

        except Exception as e:
            logger.error(f"Error when generating next question: {e}")
            return None

    @classmethod
    def _extract_questions(cls, text: str) -> List[str] | None:
        content_match = re.search(r"```(.*?)```", text, re.DOTALL)
        content = content_match.group(1) if content_match else None
        if not content:
            return None
        return [q.strip() for q in content.split("\n") if q.strip()]

    @classmethod
    async def run(
        cls,
        chat_history: List[ChatAPIMessage],
        response: str,
    ) -> List[str]:
        """
        Suggest the next questions that user might ask based on the chat history and the last response.
        """
        messages = [*chat_history, ChatAPIMessage(role="assistant", content=response)]
        return await cls.suggest_next_questions_all_messages(messages)
