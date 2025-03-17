import json
import asyncio
from unittest.mock import patch, MagicMock

import httpx
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.llms.qianfan import Qianfan

# The request and response messages come from:
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lqoklvr1
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t

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


mock_chat_response = {
    "id": "as-fg4g836x8n",
    "object": "chat.completion",
    "created": 1709716601,
    "result": "北京，简称“京”，古称燕京、北平，中华民族的发祥地之一，是中华人民共和国首都、直辖市、国家中心城市、超大城市，也是国务院批复确定的中国政治中心、文化中心、国际交往中心、科技创新中心，中国历史文化名城和古都之一，世界一线城市。\n\n北京被世界城市研究机构评为世界一线城市，联合国报告指出北京市人类发展指数居中国城市第二位。北京市成功举办夏奥会与冬奥会，成为全世界第一个“双奥之城”。北京有着3000余年的建城史和850余年的建都史，是全球拥有世界遗产（7处）最多的城市。\n\n北京是一个充满活力和创新精神的城市，也是中国传统文化与现代文明的交汇点。在这里，你可以看到古老的四合院、传统的胡同、雄伟的长城和现代化的高楼大厦交相辉映。此外，北京还拥有丰富的美食文化，如烤鸭、炸酱面等，以及各种传统艺术表演，如京剧、相声等。\n\n总的来说，北京是一个充满魅力和活力的城市，无论你是历史爱好者、美食家还是现代都市人，都能在这里找到属于自己的乐趣和归属感。",
    "is_truncated": False,
    "need_clear_history": False,
    "finish_reason": "normal",
    "usage": {"prompt_tokens": 2, "completion_tokens": 221, "total_tokens": 223},
}

mock_stream_chat_response = [
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089502,
        "sentence_id": 0,
        "is_end": False,
        "is_truncated": False,
        "result": "当然可以，",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089504,
        "sentence_id": 1,
        "is_end": False,
        "is_truncated": False,
        "result": "以下是一些建议的自驾游路线，它们涵盖了各种不同的风景和文化体验：\n\n1. **西安-敦煌历史文化之旅**：\n\n\n\t* 路线：西安",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089506,
        "sentence_id": 2,
        "is_end": False,
        "is_truncated": False,
        "result": " - 天水 - 兰州 - 嘉峪关 - 敦煌\n\t* 特点：此路线让您领略到中国西北的丰富历史文化。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089508,
        "sentence_id": 3,
        "is_end": False,
        "is_truncated": False,
        "result": "您可以参观西安的兵马俑、大雁塔，体验兰州的黄河风情，以及在敦煌欣赏壮丽的莫高窟。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089511,
        "sentence_id": 4,
        "is_end": False,
        "is_truncated": False,
        "result": "\n2. **海南环岛热带风情游**：\n\n\n\t* 路线：海口 - 三亚 - 陵水 - 万宁 - 文昌 - 海",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089512,
        "sentence_id": 5,
        "is_end": False,
        "is_truncated": False,
        "result": "口\n\t* 特点：海南岛是中国唯一的黎族聚居区，这里有独特的热带风情、美丽的海滩和丰富的水果。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 153, "total_tokens": 158},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089513,
        "sentence_id": 6,
        "is_end": False,
        "is_truncated": False,
        "result": "您可以在三亚享受阳光沙滩，品尝当地美食，感受海南的悠闲生活。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 153, "total_tokens": 158},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089516,
        "sentence_id": 7,
        "is_end": False,
        "is_truncated": False,
        "result": "\n3. **穿越阿里大北线**：\n\n\n\t* 路线：成都 - 广元 - 汉中 - 西安 - 延安 - 银川 -",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 153, "total_tokens": 158},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089518,
        "sentence_id": 8,
        "is_end": False,
        "is_truncated": False,
        "result": " 阿拉善左旗 - 额济纳旗 - 嘉峪关 - 敦煌\n\t* 特点：这是一条充满挑战的自驾路线，穿越了中国",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 153, "total_tokens": 158},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089519,
        "sentence_id": 9,
        "is_end": False,
        "is_truncated": False,
        "result": "的西部。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089519,
        "sentence_id": 10,
        "is_end": False,
        "is_truncated": False,
        "result": "您将经过壮观的沙漠、神秘的戈壁和古老的丝绸之路遗址。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089520,
        "sentence_id": 11,
        "is_end": False,
        "is_truncated": False,
        "result": "此路线适合喜欢探险和寻求不同体验的旅行者。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089523,
        "sentence_id": 12,
        "is_end": False,
        "is_truncated": False,
        "result": "\n4. **寻找北方净土 - 阿尔山自驾之旅**：\n\n\n\t* 路线：北京 - 张家口 - 张北 - 太仆寺旗",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089525,
        "sentence_id": 13,
        "is_end": False,
        "is_truncated": False,
        "result": " - 锡林浩特 - 东乌珠穆沁旗 - 满都湖宝拉格 - 宝格达林场 - 五岔沟 - 阿尔山 -",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089527,
        "sentence_id": 14,
        "is_end": False,
        "is_truncated": False,
        "result": " 伊尔施 - 新巴尔虎右旗 - 满洲里 - 北京\n\t* 特点：此路线带您穿越中国北方的草原和森林，抵达",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089527,
        "sentence_id": 15,
        "is_end": False,
        "is_truncated": False,
        "result": "风景如画的阿尔山。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089528,
        "sentence_id": 16,
        "is_end": False,
        "is_truncated": False,
        "result": "您可以在这里欣赏壮丽的自然风光，体验当地的民俗文化，享受宁静的乡村生活。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089529,
        "sentence_id": 17,
        "is_end": False,
        "is_truncated": False,
        "result": "\n\n以上路线仅供参考，您可以根据自己的兴趣和时间安排进行调整。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089531,
        "sentence_id": 18,
        "is_end": False,
        "is_truncated": False,
        "result": "在规划自驾游时，请务必注意道路安全、车辆保养以及当地的天气和交通状况。",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089531,
        "sentence_id": 19,
        "is_end": False,
        "is_truncated": False,
        "result": "祝您旅途愉快！",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 239, "total_tokens": 244},
    },
    {
        "id": "as-vb0m37ti8y",
        "object": "chat.completion",
        "created": 1709089531,
        "sentence_id": 20,
        "is_end": True,
        "is_truncated": False,
        "result": "",
        "need_clear_history": False,
        "finish_reason": "normal",
        "usage": {"prompt_tokens": 5, "completion_tokens": 420, "total_tokens": 425},
    },
]


@patch("httpx.Client")
def test_from_model_name(mock_client: httpx.Client):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_service_list_reponse
    mock_client.return_value.__enter__.return_value.send.return_value = mock_response

    llm = Qianfan.from_model_name(
        "mock_access_key", "mock_secret_key", "ERNIE-Bot 4.0", "8192"
    )
    assert llm.model_name == "ERNIE-Bot 4.0"
    assert (
        llm.endpoint_url
        == "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
    )
    assert llm.llm_type == "chat"

    mock_client.return_value.__enter__.return_value.send.assert_called_once()


@patch("httpx.AsyncClient")
def test_afrom_model_name(mock_client: httpx.AsyncClient):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_service_list_reponse
    mock_client.return_value.__aenter__.return_value.send.return_value = mock_response

    async def async_process():
        llm = await Qianfan.afrom_model_name(
            "mock_access_key", "mock_secret_key", "ERNIE-Bot 4.0", "8192"
        )
        assert llm.model_name == "ERNIE-Bot 4.0"
        assert (
            llm.endpoint_url
            == "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
        )
        assert llm.llm_type == "chat"

    asyncio.run(async_process())

    mock_client.return_value.__aenter__.return_value.send.assert_called_once()


@patch("httpx.Client")
def test_chat(mock_client: httpx.Client):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_chat_response
    mock_client.return_value.__enter__.return_value.send.return_value = mock_response

    llm = Qianfan(
        "mock_access_key",
        "mock_secret_key",
        "test-model",
        "https://127.0.0.1/test",
        8192,
    )
    resp = llm.chat([ChatMessage(role=MessageRole.USER, content="介绍一下北京")])
    assert resp.message.content == mock_chat_response["result"]

    mock_client.return_value.__enter__.return_value.send.assert_called_once()


@patch("httpx.AsyncClient")
def test_achat(mock_client: httpx.AsyncClient):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_chat_response
    mock_client.return_value.__aenter__.return_value.send.return_value = mock_response

    async def async_process():
        llm = Qianfan(
            "mock_access_key",
            "mock_secret_key",
            "test-model",
            "https://127.0.0.1/test",
            8192,
        )
        resp = await llm.achat(
            [ChatMessage(role=MessageRole.USER, content="介绍一下北京")]
        )
        assert resp.message.content == mock_chat_response["result"]

    asyncio.run(async_process())

    mock_client.return_value.__aenter__.return_value.send.assert_called_once()


@patch("httpx.Client")
def test_stream_chat(mock_client: httpx.Client):
    reply_data = ["data: " + json.dumps(item) for item in mock_stream_chat_response]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(reply_data)
    mock_client.return_value.__enter__.return_value.send.return_value = mock_response

    llm = Qianfan(
        "mock_access_key",
        "mock_secret_key",
        "test-model",
        "https://127.0.0.1/test",
        8192,
    )
    resp = llm.stream_chat(
        [ChatMessage(role=MessageRole.USER, content="给我推荐一些自驾游路线")]
    )
    last_content = ""
    content = ""
    for part in resp:
        content += part.delta
        last_content = part.message.content
    assert last_content == content
    assert last_content == "".join(
        [mock_part["result"] for mock_part in mock_stream_chat_response]
    )

    mock_client.return_value.__enter__.return_value.send.assert_called_once()


@patch("httpx.AsyncClient")
def test_astream_chat(mock_client: httpx.AsyncClient):
    reply_data = ["data: " + json.dumps(item) for item in mock_stream_chat_response]

    async def mock_async_gen():
        for part in reply_data:
            yield part

    mock_response = MagicMock()
    mock_response.aiter_lines.return_value = mock_async_gen()
    mock_client.return_value.__aenter__.return_value.send.return_value = mock_response

    async def async_process():
        llm = Qianfan(
            "mock_access_key",
            "mock_secret_key",
            "test-model",
            "https://127.0.0.1/test",
            8192,
        )
        resp = await llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="给我推荐一些自驾游路线")]
        )
        last_content = ""
        content = ""
        async for part in resp:
            content += part.delta
            last_content = part.message.content
        assert last_content == content
        assert last_content == "".join(
            [mock_part["result"] for mock_part in mock_stream_chat_response]
        )

    asyncio.run(async_process())

    mock_client.return_value.__aenter__.return_value.send.assert_called_once()
