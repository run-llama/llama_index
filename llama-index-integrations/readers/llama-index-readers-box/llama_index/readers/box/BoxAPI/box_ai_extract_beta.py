from enum import Enum

from typing import Optional

from box_sdk_gen.internal.base_object import BaseObject

from typing import List

from typing import Dict

from box_sdk_gen.serialization.json.serializer import serialize

from box_sdk_gen.serialization.json.serializer import deserialize

from box_sdk_gen.schemas.ai_response import AiResponse

from box_sdk_gen.networking.auth import Authentication

from box_sdk_gen.networking.network import NetworkSession

from box_sdk_gen.internal.utils import prepare_params

from box_sdk_gen.networking.fetch import FetchOptions

from box_sdk_gen.networking.fetch import FetchResponse

from box_sdk_gen.networking.fetch import fetch


class CreateAiExtractItemsTypeField(str, Enum):
    FILE = "file"


class CreateAiExtractItems(BaseObject):
    _discriminator = "type", {"file"}

    def __init__(
        self,
        id: str,
        *,
        type: CreateAiExtractItemsTypeField = CreateAiExtractItemsTypeField.FILE.value,
        content: Optional[str] = None,
        **kwargs,
    ):
        """
        :param id: The id of the item.
        :type id: str
        :param type: The type of the item., defaults to CreateAiAskItemsTypeField.FILE.value
        :type type: CreateAiAskItemsTypeField, optional
        :param content: The content of the item, often the text representation., defaults to None
        :type content: Optional[str], optional
        """
        super().__init__(**kwargs)
        self.id = id
        self.type = type
        self.content = content


class AiExtractManager:
    def __init__(
        self,
        *,
        auth: Optional[Authentication] = None,
        network_session: NetworkSession = None,
    ):
        if network_session is None:
            network_session = NetworkSession()
        self.auth = auth
        self.network_session = network_session

    def create_ai_extract(
        self,
        prompt: str,
        items: List[CreateAiExtractItems],
        *,
        extra_headers: Optional[Dict[str, Optional[str]]] = None,
    ) -> AiResponse:
        """
                Sends an AI request to supported LLMs and returns an answer specifically focused on the user's data structure given the provided context.
                :param prompt: The prompt provided by the client to be answered by the LLM. The prompt's length is limited to 10000 characters.
                :type prompt: str
                :param items: The items to be processed by the LLM, often files.

        **Note**: Box AI handles documents with text representations up to 1MB in size, or a maximum of 25 files, whichever comes first.
        If the file size exceeds 1MB, the first 1MB of text representation will be processed.
        If you set `mode` parameter to `single_item_qa`, the `items` array can have one element only.
                :type items: List[CreateAiAskItems]
                :param extra_headers: Extra headers that will be included in the HTTP request., defaults to None
                :type extra_headers: Optional[Dict[str, Optional[str]]], optional
        """
        if extra_headers is None:
            extra_headers = {}
        request_body: Dict = {"prompt": prompt, "items": items}
        headers_map: Dict[str, str] = prepare_params({**extra_headers})
        response: FetchResponse = fetch(
            f"{self.network_session.base_urls.base_url}/2.0/ai/extract",
            FetchOptions(
                method="POST",
                headers=headers_map,
                data=serialize(request_body),
                content_type="application/json",
                response_format="json",
                auth=self.auth,
                network_session=self.network_session,
            ),
        )
        return deserialize(response.data, AiResponse)
