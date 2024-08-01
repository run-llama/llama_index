import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import List, Optional

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class StackOverflowPost:
    link: str
    score: int
    last_activity_date: int
    creation_date: int
    post_id: Optional[int] = None
    post_type: Optional[str] = None
    body_markdown: Optional[str] = None
    owner_account_id: Optional[int] = None
    owner_reputation: Optional[int] = None
    owner_user_id: Optional[int] = None
    owner_user_type: Optional[str] = None
    owner_profile_image: Optional[str] = None
    owner_display_name: Optional[str] = None
    owner_link: Optional[str] = None
    title: Optional[str] = None
    last_edit_date: Optional[str] = None
    tags: Optional[List[str]] = None
    view_count: Optional[int] = None
    article_id: Optional[int] = None
    article_type: Optional[str] = None


def rate_limit(*, allowed_per_second: int):
    max_period = 1.0 / allowed_per_second
    last_call = [time.perf_counter()]
    lock = threading.Lock()

    def decorate(func):
        @wraps(func)
        def limit(*args, **kwargs):
            with lock:
                elapsed = time.perf_counter() - last_call[0]
                hold = max_period - elapsed
                if hold > 0:
                    time.sleep(hold)
                result = func(*args, **kwargs)
                last_call[0] = time.perf_counter()
            return result

        return limit

    return decorate


@rate_limit(allowed_per_second=15)
def rate_limited_get(url, headers):
    """
    https://api.stackoverflowteams.com/docs/throttle
    https://api.stackexchange.com/docs/throttle
    Every application is subject to an IP based concurrent request throttle.
    If a single IP is making more than 30 requests a second, new requests will be dropped.
    The exact ban period is subject to change, but will be on the order of 30 seconds to a few minutes typically.
    Note that exactly what response an application gets (in terms of HTTP code, text, and so on)
    is undefined when subject to this ban; we consider > 30 request/sec per IP to be very abusive and thus cut the requests off very harshly.
    """
    resp = requests.get(url, headers=headers)
    if resp.status_code == 429:
        logger.warning("Rate limited, sleeping for 5 minutes")
        time.sleep(300)
        return rate_limited_get(url, headers)
    return resp


class StackoverflowReader(BaseReader):
    def __init__(
        self, api_key: str = None, team_name: str = None, cache_dir: str = None
    ) -> None:
        self._api_key = api_key or os.environ.get("STACKOVERFLOW_PAT")
        self._team_name = team_name or os.environ.get("STACKOVERFLOW_TEAM_NAME")
        self._last_index_time = None  # TODO
        self._cache_dir = cache_dir
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)

    def load_data(
        self, page: int = 1, doc_type: str = "posts", limit: int = 50
    ) -> List[Document]:
        data = []
        has_more = True

        while has_more:
            url = self.build_url(page, doc_type)
            headers = {"X-API-Access-Token": self._api_key}
            fp = os.path.join(self._cache_dir, f"{doc_type}_{page}.json")
            response = {}
            if self._cache_dir and os.path.exists(fp) and os.path.getsize(fp) > 0:
                try:
                    with open(fp) as f:
                        response = f.read()
                        response = json.loads(response)
                except Exception as e:
                    logger.error(e)
            if not response:
                response = rate_limited_get(url, headers)
                response.raise_for_status()
                if self._cache_dir:
                    with open(
                        os.path.join(self._cache_dir, f"{doc_type}_{page}.json"), "w"
                    ) as f:
                        f.write(response.content.decode("utf-8"))
                    logger.info(f"Wrote {fp} to cache")
                response = response.json()
            has_more = response["has_more"]
            items = response["items"]
            logger.info(f"Fetched {len(items)} {doc_type} from Stack Overflow")

            for item_dict in items:
                owner_fields = {}
                if "owner" in item_dict:
                    owner_fields = {
                        f"owner_{k}": v for k, v in item_dict.pop("owner").items()
                    }
                if "title" not in item_dict:
                    item_dict["title"] = item_dict["link"]
                post = StackOverflowPost(**item_dict, **owner_fields)
                # TODO: filter out old posts
                # last_modified = datetime.fromtimestamp(post.last_edit_date or post.last_activity_date)
                # if last_modified < self._last_index_time:
                #     return data

                post_document = Document(
                    text=post.body_markdown,
                    doc_id=post.post_id,
                    extra_info={
                        "title": post.title,
                        "author": post.owner_display_name,
                        "timestamp": datetime.fromtimestamp(post.creation_date),
                        "location": post.link,
                        "url": post.link,
                        "author_image_url": post.owner_profile_image,
                        "type": post.post_type,
                    },
                )
                data.append(post_document)

            if has_more:
                page += 1

        return data

    def build_url(self, page: int, doc_type: str) -> str:
        team_fragment = f"&team={self._team_name}"
        # not sure if this filter is shared globally, or only to a particular team
        filter_fragment = "&filter=!nOedRLbqzB"
        page_fragment = f"&page={page}"
        return f"https://api.stackoverflowteams.com/2.3/{doc_type}?{team_fragment}{filter_fragment}{page_fragment}"


if __name__ == "__main__":
    reader = StackoverflowReader(
        os.environ.get("STACKOVERFLOW_PAT"),
        os.environ.get("STACKOVERFLOW_TEAM_NAME"),
        cache_dir="./stackoverflow_cache",
    )
    # reader.load_data()
