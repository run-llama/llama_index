import requests


def post(base_url, headers, params):
    response = requests.post(base_url, headers=headers, json=params)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    response_dict = response.json()
    if response_dict["code"] != "Success":
        raise RuntimeError(response_dict)
    return response_dict


def get(base_url, headers, params):
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code != 200:
        raise RuntimeError(response.text)

    response_dict = response.json()
    if response_dict["code"] != "Success":
        raise RuntimeError(response_dict)
    return response_dict


def get_pipeline_id(base_url, headers, params):
    response_dict = get(base_url, headers, params)
    return response_dict.get("id", "")
