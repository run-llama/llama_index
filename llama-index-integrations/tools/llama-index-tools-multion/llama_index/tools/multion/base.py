"""Multion tool spec."""

import base64
from io import BytesIO
from typing import Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MultionToolSpec(BaseToolSpec):
    """Multion tool spec."""

    spec_functions = ["browse"]

    def __init__(self, token_file: Optional[str] = "multion_token.txt") -> None:
        """Initialize with parameters."""
        import multion

        multion.login()
        self.last_tab = None

    def browse(self, instruction: str):
        """
        Browse the web using Multion
        Multion gives the ability for LLMs to control web browsers using natural language instructions.

        You may have to repeat the instruction through multiple steps or update your instruction to get to
        the final desired state. If the status is 'CONTINUE', reissue the same instruction to continue execution

        Args:
            instruction (str): The detailed and specific natural language instructrion for web browsing
        """
        import multion

        if self.last_tab:
            session = multion.update_session(self.last_tab, {"input": instruction})
        else:
            session = multion.new_session(
                {"input": instruction, "url": "https://google.com"}
            )
            self.last_tab = session["tabId"]

        return {
            "url": session["url"],
            "status": session["status"],
            "action_completed": session["message"],
            "content": self._read_screenshot(session["screenshot"]),
        }

    def _read_screenshot(self, screenshot) -> str:
        import pytesseract
        from PIL import Image

        image_bytes = screenshot.replace("data:image/png;base64,", "")
        image = Image.open(self._bytes_to_image(image_bytes))

        return pytesseract.image_to_string(image)

    def _bytes_to_image(self, img_bytes):
        return BytesIO(base64.b64decode(img_bytes))
