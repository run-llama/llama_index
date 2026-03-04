"""Rayyan review reader."""

import logging
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class RayyanReader(BaseReader):
    """
    Rayyan reader. Reads articles from a Rayyan review.

    Args:
        credentials_path (str): Rayyan credentials path.
        rayyan_url (str, optional): Rayyan URL. Defaults to https://rayyan.ai.
            Set to an alternative URL if you are using a non-production Rayyan instance.

    """

    def __init__(
        self, credentials_path: str, rayyan_url: str = "https://rayyan.ai"
    ) -> None:
        """Initialize Rayyan reader."""
        from rayyan import Rayyan
        from rayyan.user import User

        logging.debug("Initializing Rayyan reader...")
        self.rayyan = Rayyan(credentials_path, url=rayyan_url)
        user = User(self.rayyan).get_info()
        logging.info(f"Signed in successfully to Rayyan as: {user['displayName']}!")

    def load_data(self, review_id: str, filters: dict = {}) -> List[Document]:
        """
        Load articles from a review.

        Args:
            review_id (int): Rayyan review ID.
            filters (dict, optional): Filters to apply to the review. Defaults to None. Passed to
                the Rayyan review results method as is.


        Returns:
            List[Document]: List of documents.

        """
        from tenacity import (
            retry,
            stop_after_attempt,
            stop_after_delay,
            stop_all,
            wait_random_exponential,
        )
        from tqdm import tqdm

        from rayyan.review import Review

        rayyan_review = Review(self.rayyan)
        my_review = rayyan_review.get(review_id)
        logging.info(
            f"Working on review: '{my_review['title']}' with {my_review['total_articles']} total articles."
        )

        result_params = {"start": 0, "length": 100}
        result_params.update(filters)

        @retry(
            wait=wait_random_exponential(min=1, max=10),
            stop=stop_all(stop_after_attempt(3), stop_after_delay(30)),
        )
        def fetch_results_with_retry():
            logging.debug("Fetch parameters: %s", result_params)
            return rayyan_review.results(review_id, result_params)

        articles = []
        logging.info("Fetching articles from Rayyan...")
        total = my_review["total_articles"]
        with tqdm(total=total) as pbar:
            while len(articles) < total:
                # retrieve articles in batches
                review_results = fetch_results_with_retry()
                fetched_articles = review_results["data"]
                articles.extend(fetched_articles)
                # update total in case filters are applied
                if total != review_results["recordsFiltered"]:
                    total = review_results["recordsFiltered"]
                    pbar.total = total
                result_params["start"] += len(fetched_articles)
                pbar.update(len(fetched_articles))

        results = []
        for article in articles:
            # iterate over all abstracts
            abstracts = ""
            if article["abstracts"] is not None:
                abstracts_arr = [
                    abstract["content"] for abstract in article["abstracts"]
                ]
                if len(abstracts_arr) > 0:
                    # map array into a string
                    abstracts = "\n".join(abstracts_arr)[0:1024].strip()
            title = article["title"]
            if title is not None:
                title = title.strip()
            body = f"{title}\n{abstracts}"
            if body.strip() == "":
                continue
            extra_info = {"id": article["id"], "title": title}

            results.append(
                Document(
                    text=body,
                    extra_info=extra_info,
                )
            )

        return results
