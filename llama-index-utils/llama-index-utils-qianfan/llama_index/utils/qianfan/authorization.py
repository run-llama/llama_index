import hmac
import hashlib
import urllib.parse
from typing import List, Dict, Tuple
from datetime import datetime, timezone
import urllib.parse


def encode_canonical_query(query: str) -> str:
    """
    Encoding the HTTP query.
    """
    parsed = urllib.parse.parse_qs(query, keep_blank_values=True)

    items: List[str] = []
    for key, values in parsed.items():
        encoded_key = urllib.parse.quote_plus(key)
        if key.lower() == "authorization":
            continue
        if len(values) > 1:  # multi value
            for val in values:
                item = encoded_key + "=" + urllib.parse.quote_plus(val)
                items.append(item)
        elif len(values[0]) > 0:  # single value
            item = encoded_key + "=" + urllib.parse.quote_plus(values[0])
            items.append(item)
        else:  # just key, no value
            item = encoded_key + "="
            items.append(item)

    items = sorted(items)
    return "&".join(items)


def encode_canonical_headers(headers: Dict[str, str], host: str) -> Tuple[str, str]:
    """
    Encoding the HTTTP headers.
    """
    new_headers: Dict[str, str] = {}
    for key, value in headers.items():
        key = key.lower()
        new_headers[key] = value
    headers = new_headers

    if "host" not in headers:
        headers["host"] = host.strip()

    signed_headers: List[str] = []
    canonical_headers: List[str] = []
    for key, value in headers.items():
        if key.find("x-bce-") != 0 and key not in (
            "host",
            "content-length",
            "content-type",
            "content-md5",
        ):
            continue
        signed_headers.append(key)

        if value != "":
            header = urllib.parse.quote_plus(key) + ":" + urllib.parse.quote_plus(value)
            canonical_headers.append(header)
    signed_headers = sorted(signed_headers)
    canonical_headers = sorted(canonical_headers)

    return ";".join(signed_headers), "\n".join(canonical_headers)


def encode_authorization(
    method: str, url: str, headers: Dict[str, str], access_key: str, secret_key: str
) -> str:
    """
    Compute the signature for the API.
    Document: https://cloud.baidu.com/doc/Reference/s/Njwvz1wot .

    :param method: HTTP method.
    :param url: HTTP URL with query string.
    :param headers: HTTP headers.
    :param access_key: The Access Key obtained from the Security Authentication Center of Baidu Intelligent Cloud Console.
    :param secret_key: The Secret Key paired with the Access Key.
    :return: The Authorization value.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    expire_in_seconds = 60

    url_parsed = urllib.parse.urlparse(url)

    auth_string_prefix = f"bce-auth-v1/{access_key}/{timestamp}/{expire_in_seconds}"

    canonical_url = urllib.parse.quote(url_parsed.path)
    canonical_query = encode_canonical_query(url_parsed.query)
    signed_headers, canonical_headers = encode_canonical_headers(
        headers, url_parsed.hostname
    )
    canonical_request = (
        method.upper()
        + "\n"
        + canonical_url
        + "\n"
        + canonical_query
        + "\n"
        + canonical_headers
    )

    signing_key = hmac.new(
        secret_key.encode(), auth_string_prefix.encode(), hashlib.sha256
    ).hexdigest()
    signature = hmac.new(
        signing_key.encode(), canonical_request.encode(), hashlib.sha256
    ).hexdigest()

    return f"bce-auth-v1/{access_key}/{timestamp}/{expire_in_seconds}/{signed_headers}/{signature}"
