import json
import os
import requests
from typing import List, Optional, Any, Iterable
from pydantic import BaseModel
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

# Default Field Mask
FIELD_MASK = "places.name,places.id,places.types,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.displayName,places.reviews,places.photos,nextPageToken"
# Default number of results
DEFAULT_NUMBER_OF_RESULTS = 100
# Search text URL
SEARCH_TEXT_BASE_URL = "https://places.googleapis.com/v1/places:searchText"
# Maximum results per page
MAX_RESULTS_PER_PAGE = 20


class Review(BaseModel):
    author_name: str
    rating: int
    text: str
    relative_publish_time: str


class Place(BaseModel):
    reviews: List[Review]
    address: str
    average_rating: float
    display_name: str
    number_of_ratings: int


class GoogleMapsTextSearchReader(BaseReader):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in the environment variables as 'GOOGLE_MAPS_API_KEY'"
            )

    def load_data(
        self,
        text: str,
        number_of_results: Optional[int] = DEFAULT_NUMBER_OF_RESULTS,
    ) -> List[Document]:
        """
        Load data from Google Maps.

        Args:
            text (str): the text to search for.
            number_of_results (Optional[int]): the number of results to return. Defaults to 20.

        """
        response = self._search_text_request(text, MAX_RESULTS_PER_PAGE)
        documents = []
        while "nextPageToken" in response:
            next_page_token = response["nextPageToken"]
            places = response.get("places", [])
            if len(places) == 0:
                break
            for place in places:
                formatted_address = place["formattedAddress"]
                average_rating = place["rating"]
                display_name = place["displayName"]
                if isinstance(display_name, dict):
                    display_name = display_name["text"]
                number_of_ratings = place["userRatingCount"]
                reviews = []
                for review in place["reviews"]:
                    review_text = review["text"]["text"]
                    author_name = review["authorAttribution"]["displayName"]
                    relative_publish_time = review["relativePublishTimeDescription"]
                    rating = review["rating"]
                    reviews.append(
                        Review(
                            author_name=author_name,
                            rating=rating,
                            text=review_text,
                            relative_publish_time=relative_publish_time,
                        )
                    )

                place = Place(
                    reviews=reviews,
                    address=formatted_address,
                    average_rating=average_rating,
                    display_name=display_name,
                    number_of_ratings=number_of_ratings,
                )
                reviews_text = "\n".join(
                    [
                        f"Author: {review.author_name}, Rating: {review.rating}, Text: {review.text}, Relative Publish Time: {review.relative_publish_time}"
                        for review in reviews
                    ]
                )
                place_text = f"Place: {place.display_name}, Address: {place.address}, Average Rating: {place.average_rating}, Number of Ratings: {place.number_of_ratings}"
                document_text = f"{place_text}\n{reviews_text}"

                if len(documents) == number_of_results:
                    return documents

                documents.append(Document(text=document_text, extra_info=place.dict()))
            response = self._search_text_request(
                text, MAX_RESULTS_PER_PAGE, next_page_token
            )

        return documents

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data lazily from Google Maps."""
        yield from self.load_data(*args, **load_kwargs)

    def _search_text_request(
        self, text, number_of_results, next_page_token: Optional[str] = None
    ) -> dict:
        """
        Send a request to the Google Maps Places API to search for text.

        Args:
            text (str): the text to search for.
            number_of_results (int): the number of results to return.
            next_page_token (Optional[str]): the next page token to get the next page of results.

        """
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": FIELD_MASK,
        }
        payload = json.dumps(
            {
                "textQuery": text,
                "pageSize": number_of_results,
                "pageToken": next_page_token,
            }
        )
        response = requests.post(SEARCH_TEXT_BASE_URL, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
