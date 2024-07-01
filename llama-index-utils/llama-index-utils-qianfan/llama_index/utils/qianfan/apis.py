from typing import Sequence, Literal, List

from llama_index.core.bridge.pydantic import BaseModel, Field

from llama_index.utils.qianfan.client import Client

APIType = Literal["chat", "completions", "embeddings", "text2image", "image2text"]


class ServiceItem(BaseModel):
    """
    Model service item.
    """

    name: str
    """model name. example: ERNIE-4.0-8K"""

    url: str
    """endpoint url. example: https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"""

    api_type: APIType = Field(..., alias="apiType")
    """api type"""

    charge_status: Literal["NOTOPEN", "OPENED", "STOP", "FREE"] = Field(
        ..., alias="chargeStatus"
    )
    """Payment status"""


class ServiceListResult(BaseModel):
    """
    All model service items.
    """

    common: List[ServiceItem]
    """built-in model service"""

    custom: List[ServiceItem]
    """custom model service"""


class ServiceListResponse(BaseModel):
    """
    Response for Querying the List of Model Serving.
    """

    result: ServiceListResult
    """All model available service items."""


def get_service_list(
    access_key: str, secret_key: str, api_type_filter: Sequence[APIType] = []
):
    """
    Get a list of available model services. Can be filtered by api type.
    """
    url = "https://qianfan.baidubce.com/wenxinworkshop/service/list"
    json = {"apiTypefilter": api_type_filter}

    client = Client(access_key, secret_key)
    resp_dict = client.post(url, json=json)
    resp = ServiceListResponse(**resp_dict)

    common_services = filter(
        lambda service: service.charge_status == "OPENED", resp.result.common
    )
    custom_services = filter(
        lambda service: service.charge_status == "OPENED", resp.result.custom
    )
    return list(common_services) + list(custom_services)


async def aget_service_list(
    access_key: str, secret_key: str, api_type_filter: Sequence[APIType] = []
):
    """
    Asynchronous get a list of available model services. Can be filtered by api type.
    """
    url = "https://qianfan.baidubce.com/wenxinworkshop/service/list"
    json = {"apiTypefilter": api_type_filter}

    client = Client(access_key, secret_key)
    resp_dict = await client.apost(url, json=json)
    resp = ServiceListResponse(**resp_dict)

    common_services = filter(
        lambda service: service.charge_status == "OPENED", resp.result.common
    )
    custom_services = filter(
        lambda service: service.charge_status == "OPENED", resp.result.custom
    )
    return list(common_services) + list(custom_services)
