import asyncio
from unittest.mock import patch, MagicMock
from typing import List

import httpx

from llama_index.utils.qianfan.apis import (
    get_service_list,
    aget_service_list,
    ServiceItem,
)

mock_service_list_reponse = {
    "log_id": "4102908182",
    "success": True,
    "result": {
        "common": [
            {
                "name": "ERNIE-Bot 4.0",
                "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro",
                "apiType": "chat",
                "chargeStatus": "OPENED",
                "versionList": [{"trainType": "ernieBot_4", "serviceStatus": "Done"}],
            }
        ],
        "custom": [
            {
                "serviceId": "123",
                "serviceUuid": "svco-xxxxaaa",
                "name": "conductor_liana2",
                "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ca6zisxxxx",
                "apiType": "chat",
                "chargeStatus": "NOTOPEN",
                "versionList": [
                    {
                        "aiModelId": "xxx-123",
                        "aiModelVersionId": "xxx-456",
                        "trainType": "llama2-7b",
                        "serviceStatus": "Done",
                    }
                ],
            }
        ],
    },
}


@patch("httpx.Client")
def test_get_service_list(mock_client: httpx.Client):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_service_list_reponse
    mock_client.return_value.__enter__.return_value.send.return_value = mock_response

    service_list: List[ServiceItem] = get_service_list(
        "mock_access_key", "mock_secret_key", api_type_filter=["chat"]
    )
    assert len(service_list) == 1  # Only return models with the status OPENED.
    assert service_list[0].name == "ERNIE-Bot 4.0"
    assert (
        service_list[0].url
        == "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
    )
    assert service_list[0].api_type == "chat"
    assert service_list[0].charge_status == "OPENED"

    mock_client.return_value.__enter__.return_value.send.assert_called_once()


@patch("httpx.AsyncClient")
def test_aget_service_list(mock_client: httpx.AsyncClient):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_service_list_reponse
    mock_client.return_value.__aenter__.return_value.send.return_value = mock_response

    async def async_process():
        service_list: List[ServiceItem] = await aget_service_list(
            "mock_access_key", "mock_secret_key", api_type_filter=["chat"]
        )
        # Only return models with the status OPENED.
        assert len(service_list) == 1
        assert service_list[0].name == "ERNIE-Bot 4.0"
        assert (
            service_list[0].url
            == "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
        )
        assert service_list[0].api_type == "chat"
        assert service_list[0].charge_status == "OPENED"

    asyncio.run(async_process())

    mock_client.return_value.__aenter__.return_value.send.assert_called_once()
