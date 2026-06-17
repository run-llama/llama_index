from typing import Optional

import requests
import time


def post(base_url: str, headers: dict, params: dict, timeout: Optional[float] = 60.0):
    response = requests.post(base_url, headers=headers, json=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    response_dict = response.json()
    if response_dict["code"] != "Success":
        raise RuntimeError(response_dict)
    return response_dict


def get(base_url: str, headers: dict, params: dict, timeout: Optional[float] = 60.0):
    response = requests.get(base_url, headers=headers, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(response.text)

    response_dict = response.json()
    if response_dict["code"] != "Success":
        raise RuntimeError(response_dict)
    return response_dict


def get_pipeline_id(
    base_url: str, headers: dict, params: dict, timeout: Optional[float] = 60.0
):
    response_dict = get(base_url, headers, params, timeout)
    return response_dict.get("id", "")


def run_ingestion(
    request_url: str,
    headers: dict,
    verbose: bool = False,
    timeout: Optional[float] = 60.0,
):
    ingestion_status = ""
    failed_docs = []

    while True:
        response = requests.get(request_url, headers=headers, timeout=timeout)
        try:
            response_text = response.json()
        except Exception as e:
            print(f"Failed to get response: \n{response.text}\nretrying...")
            continue

        if response_text.get("code", "") != "Success":
            print(
                f"Failed to get ingestion status: {response_text.get('message', '')}\n{response_text}\nretrying..."
            )
            continue
        ingestion_status = response_text.get("ingestion_status", "")
        failed_docs = response_text.get("failed_docs", "")
        if verbose:
            print(f"Current status: {ingestion_status}")
        if ingestion_status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(5)
    return ingestion_status, failed_docs
