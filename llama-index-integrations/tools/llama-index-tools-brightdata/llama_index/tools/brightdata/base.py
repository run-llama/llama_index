"""Bright Data tool spec for LlamaIndex."""

from typing import Dict, Optional
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class BrightDataToolSpec(BaseToolSpec):
    """Bright Data tool spec for web scraping and search capabilities."""

    spec_functions = [
        "scrape_as_markdown",
        "get_screenshot",
        "search_engine",
        "web_data_feed",
    ]

    def __init__(
        self,
        api_key: str,
        zone: str = "unblocker",
        verbose: bool = False,
    ) -> None:
        """
        Initialize with API token and default zone.

        Args:
            api_key (str): Your Bright Data API token
            zone (str): Bright Data zone name
            verbose (bool): Print additional information about requests

        """
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self._api_key = api_key
        self._zone = zone
        self._verbose = verbose
        self._endpoint = "https://api.brightdata.com/request"

    def _make_request(self, payload: Dict) -> str:
        """
        Make a request to Bright Data API.

        Args:
            payload (Dict): Request payload

        Returns:
            str: Response text

        """
        import requests
        import json

        if self._verbose:
            print(f"[Bright Data] Request: {payload['url']}")

        response = requests.post(
            self._endpoint, headers=self._headers, data=json.dumps(payload)
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to scrape: {response.status_code} - {response.text}"
            )

        return response.text

    def scrape_as_markdown(self, url: str, zone: Optional[str] = None) -> Document:
        """
        Scrape a webpage and return content in Markdown format.

        Args:
            url (str): URL to scrape
            zone (Optional[str]): Override default zone

        Returns:
            Document: Scraped content as Markdown

        """
        payload = {
            "url": url,
            "zone": zone or self._zone,
            "format": "raw",
            "data_format": "markdown",
        }

        content = self._make_request(payload)
        return Document(text=content, metadata={"url": url})

    def get_screenshot(
        self, url: str, output_path: str, zone: Optional[str] = None
    ) -> str:
        """
        Take a screenshot of a webpage.

        Args:
            url (str): URL to screenshot
            output_path (str): Path to save the screenshot
            zone (Optional[str]): Override default zone

        Returns:
            str: Path to saved screenshot

        """
        import requests
        import json

        payload = {
            "url": url,
            "zone": zone or self._zone,
            "format": "raw",
            "data_format": "screenshot",
        }

        response = requests.post(
            self._endpoint, headers=self._headers, data=json.dumps(payload)
        )

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    def search_engine(
        self,
        query: str,
        engine: str = "google",
        zone: Optional[str] = None,
        language: Optional[str] = None,  # hl parameter, e.g., "en"
        country_code: Optional[str] = None,  # gl parameter, e.g., "us"
        search_type: Optional[
            str
        ] = None,  # tbm parameter (images, shopping, news, etc.)
        start: Optional[int] = None,  # pagination start index
        num_results: Optional[int] = 10,  # number of results to return
        location: Optional[str] = None,  # uule parameter for geo-location
        device: Optional[str] = None,  # device type for user-agent
        return_json: bool = False,  # parse results as JSON
        hotel_dates: Optional[str] = None,  # check-in and check-out dates
        hotel_occupancy: Optional[int] = None,  # number of guests
    ) -> Document:
        """
        Search using Google, Bing, or Yandex with advanced parameters and return results in Markdown.

        Args:
            query (str): Search query
            engine (str): Search engine - 'google', 'bing', or 'yandex'
            zone (Optional[str]): Override default zone

            # Google SERP specific parameters
            language (Optional[str]): Two-letter language code (hl parameter)
            country_code (Optional[str]): Two-letter country code (gl parameter)
            search_type (Optional[str]): Type of search (images, shopping, news, etc.)
            start (Optional[int]): Results pagination offset (0=first page, 10=second page)
            num_results (Optional[int]): Number of results to return (default 10)
            location (Optional[str]): Location for search results (uule parameter)
            device (Optional[str]): Device type (mobile, ios, android, ipad, android_tablet)
            return_json (bool): Return parsed JSON instead of HTML/Markdown

            # Hotel search parameters
            hotel_dates (Optional[str]): Check-in and check-out dates (format: YYYY-MM-DD,YYYY-MM-DD)
            hotel_occupancy (Optional[int]): Number of guests (1-4)

        Returns:
            Document: Search results as Markdown or JSON

        """
        encoded_query = self._encode_query(query)

        base_urls = {
            "google": f"https://www.google.com/search?q={encoded_query}",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
            "yandex": f"https://yandex.com/search/?text={encoded_query}",
        }

        if engine not in base_urls:
            raise ValueError(
                f"Unsupported search engine: {engine}. Use 'google', 'bing', or 'yandex'"
            )

        search_url = base_urls[engine]

        if engine == "google":
            params = []

            if language:
                params.append(f"hl={language}")

            if country_code:
                params.append(f"gl={country_code}")

            if search_type:
                if search_type == "jobs":
                    params.append("ibp=htl;jobs")
                else:
                    search_types = {"images": "isch", "shopping": "shop", "news": "nws"}
                    tbm_value = search_types.get(search_type, search_type)
                    params.append(f"tbm={tbm_value}")

            if start is not None:
                params.append(f"start={start}")

            if num_results:
                params.append(f"num={num_results}")

            if location:
                params.append(f"uule={self._encode_query(location)}")

            if device:
                device_value = "1"

                if device in ["ios", "iphone"]:
                    device_value = "ios"
                elif device == "ipad":
                    device_value = "ios_tablet"
                elif device == "android":
                    device_value = "android"
                elif device == "android_tablet":
                    device_value = "android_tablet"

                params.append(f"brd_mobile={device_value}")

            if return_json:
                params.append("brd_json=1")

            if hotel_dates:
                params.append(f"hotel_dates={self._encode_query(hotel_dates)}")

            if hotel_occupancy:
                params.append(f"hotel_occupancy={hotel_occupancy}")

            if params:
                search_url += "&" + "&".join(params)

        payload = {
            "url": search_url,
            "zone": zone or self._zone,
            "format": "raw",
            "data_format": "markdown" if not return_json else "raw",
        }

        content = self._make_request(payload)
        return Document(
            text=content, metadata={"query": query, "engine": engine, "url": search_url}
        )

    def web_data_feed(
        self,
        source_type: str,
        url: str,
        num_of_reviews: Optional[int] = None,
        timeout: int = 600,
        polling_interval: int = 1,
    ) -> Dict:
        """
        Retrieve structured web data from various sources like LinkedIn, Amazon, Instagram, etc.

        Args:
            source_type (str): Type of data source (e.g., 'linkedin_person_profile', 'amazon_product')
            url (str): URL of the web resource to retrieve data from
            num_of_reviews (Optional[int]): Number of reviews to retrieve (only for facebook_company_reviews)
            timeout (int): Maximum time in seconds to wait for data retrieval
            polling_interval (int): Time in seconds between polling attempts

        Returns:
            Dict: Structured data from the requested source

        """
        import requests
        import time

        datasets = {
            "amazon_product": "gd_l7q7dkf244hwjntr0",
            "amazon_product_reviews": "gd_le8e811kzy4ggddlq",
            "linkedin_person_profile": "gd_l1viktl72bvl7bjuj0",
            "linkedin_company_profile": "gd_l1vikfnt1wgvvqz95w",
            "zoominfo_company_profile": "gd_m0ci4a4ivx3j5l6nx",
            "instagram_profiles": "gd_l1vikfch901nx3by4",
            "instagram_posts": "gd_lk5ns7kz21pck8jpis",
            "instagram_reels": "gd_lyclm20il4r5helnj",
            "instagram_comments": "gd_ltppn085pokosxh13",
            "facebook_posts": "gd_lyclm1571iy3mv57zw",
            "facebook_marketplace_listings": "gd_lvt9iwuh6fbcwmx1a",
            "facebook_company_reviews": "gd_m0dtqpiu1mbcyc2g86",
            "x_posts": "gd_lwxkxvnf1cynvib9co",
            "zillow_properties_listing": "gd_lfqkr8wm13ixtbd8f5",
            "booking_hotel_listings": "gd_m5mbdl081229ln6t4a",
            "youtube_videos": "gd_m5mbdl081229ln6t4a",
        }

        if source_type not in datasets:
            valid_sources = ", ".join(datasets.keys())
            raise ValueError(
                f"Invalid source_type: {source_type}. Valid options are: {valid_sources}"
            )

        dataset_id = datasets[source_type]

        request_data = {"url": url}
        if source_type == "facebook_company_reviews" and num_of_reviews is not None:
            request_data["num_of_reviews"] = str(num_of_reviews)

        trigger_response = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            params={"dataset_id": dataset_id, "include_errors": True},
            headers=self._headers,
            json=[request_data],
        )

        trigger_data = trigger_response.json()
        if not trigger_data.get("snapshot_id"):
            raise Exception("No snapshot ID returned from trigger request")

        snapshot_id = trigger_data["snapshot_id"]
        if self._verbose:
            print(
                f"[Bright Data] {source_type} triggered with snapshot ID: {snapshot_id}"
            )

        attempts = 0
        max_attempts = timeout

        while attempts < max_attempts:
            try:
                snapshot_response = requests.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    params={"format": "json"},
                    headers=self._headers,
                )

                snapshot_data = snapshot_response.json()

                if (
                    isinstance(snapshot_data, dict)
                    and snapshot_data.get("status") == "running"
                ):
                    if self._verbose:
                        print(
                            f"[Bright Data] Snapshot not ready, polling again (attempt {attempts + 1}/{max_attempts})"
                        )
                    attempts += 1
                    time.sleep(polling_interval)
                    continue

                if self._verbose:
                    print(f"[Bright Data] Data received after {attempts + 1} attempts")

                return snapshot_data

            except Exception as e:
                if self._verbose:
                    print(f"[Bright Data] Polling error: {e!s}")
                attempts += 1
                time.sleep(polling_interval)

        raise TimeoutError(
            f"Timeout after {max_attempts} seconds waiting for {source_type} data"
        )

    @staticmethod
    def _encode_query(query: str) -> str:
        """URL encode a search query."""
        from urllib.parse import quote

        return quote(query)
