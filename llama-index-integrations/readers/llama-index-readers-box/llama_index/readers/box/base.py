import logging
from typing import List

from llama_index.core.readers.base import (
    BasePydanticReader,
)
from llama_index.core.schema import Document

from llama_index.readers.box.box_client_ccg import reader_box_client_ccg

logger = logging.getLogger(__name__)


class BoxReader(BasePydanticReader):
    box_client_id: str
    box_client_secret: str
    box_enterprise_id: str
    box_user_id: str = None
    # client: BoxClient

    @classmethod
    def class_name(cls) -> str:
        return "BoxReader"

    def load_data(self) -> List[Document]:
        client = reader_box_client_ccg(
            self.box_client_id,
            self.box_client_secret,
            self.box_enterprise_id,
            self.box_user_id,
        )
        me = client.users.get_user_me()
