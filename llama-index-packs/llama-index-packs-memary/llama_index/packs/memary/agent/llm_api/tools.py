# From MemGPT llm_api_tools.py:

import urllib
import logging
import requests


def smart_urljoin(base_url, relative_url):
    """urljoin is stupid and wants a trailing / at the end of the endpoint address, or it will chop the suffix off"""
    if not base_url.endswith("/"):
        base_url += "/"
    return urllib.parse.urljoin(base_url, relative_url)


def openai_chat_completions_request(url, api_key, data):
    """text-generation?lang=curl"""

    url = smart_urljoin(url, "chat/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    logging.info(f"Sending request to {url}")
    try:
        # Example code to trigger a rate limit response:
        # mock_response = requests.Response()
        # mock_response.status_code = 429
        # http_error = requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
        # http_error.response = mock_response
        # raise http_error

        # Example code to trigger a context overflow response (for an 8k model)
        # data["messages"][-1]["content"] = " ".join(["repeat after me this is not a fluke"] * 1000)

        response = requests.post(url, headers=headers, json=data)
        logging.info(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        logging.info(f"response.json = {response}")
        # response = ChatCompletionResponse(**response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        logging.error(f"Got HTTPError, exception={http_err}, payload={data}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        logging.warning(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        logging.warning(f"Got unknown Exception, exception={e}")
        raise e


def ollama_chat_completions_request(messages, model):
    """sends chat request to model running on Ollama"""

    url = "http://localhost:11434/api/chat"
    data = {"model": model, "messages": messages, "stream": False}

    logging.info(f"Sending request to {url}")
    try:
        response = requests.post(url, json=data)
        logging.info(f"response = {response}")
        response.raise_for_status()
        response = response.json()
        logging.info(f"response.json = {response}")
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        logging.error(f"Got HTTPError, exception={http_err}, payload={data}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        logging.warning(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        logging.warning(f"Got unknown Exception, exception={e}")
        raise e
