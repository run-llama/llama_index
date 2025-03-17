"""
MangaDex info reader.

Retrieves data about a particular manga by title.
"""

import logging
from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class MangaDexReader(BaseReader):
    def __init__(self) -> None:
        self.base_url = "https://api.mangadex.org"

    def _get_manga_info(self, title: str):
        try:
            manga_response = requests.get(
                f"{self.base_url}/manga", params={"title": title}
            )
            manga_response.raise_for_status()
            manga_data = manga_response.json()["data"]

            if len(manga_data):
                return manga_data[0]
            else:
                logger.warning(f"No match found for title '{title}'")
                return None
        except requests.exceptions.HTTPError as http_error:
            logger.error(f"HTTP error: {http_error}")
        except requests.exceptions.RequestException as req_error:
            logger.error(f"Request Error: {req_error}")
        return None

    # Authors and artists are combined
    def _get_manga_author(self, id: str):
        try:
            author_response = requests.get(
                f"{self.base_url}/author", params={"ids[]": [id]}
            )
            author_response.raise_for_status()
            return author_response.json()["data"][0]
        except requests.exceptions.HTTPError as http_error:
            logger.error(f"HTTP error: {http_error}")
        except requests.exceptions.RequestException as req_error:
            logger.error(f"Request Error: {req_error}")
        return None

    def _get_manga_chapters(self, manga_id: str, lang: str):
        try:
            chapter_response = requests.get(
                f"{self.base_url}/manga/{manga_id}/feed",
                params={
                    "translatedLanguage[]": [lang],
                    "order[chapter]": "asc",
                },
            )
            chapter_response.raise_for_status()
            return chapter_response.json()
        except requests.exceptions.HTTPError as http_error:
            logger.error(f"HTTP error: {http_error}")
        except requests.exceptions.RequestException as req_error:
            logger.error(f"Request Error: {req_error}")
        return None

    def load_data(self, titles: List[str], lang: str = "en") -> List[Document]:
        """
        Load data from the MangaDex API.

        Args:
            title (List[str]): List of manga titles
            lang (str, optional): ISO 639-1 language code. Defaults to 'en'.


        Returns:
            List[Document]: A list of Documents.

        """
        result = []
        for title in titles:
            manga = self._get_manga_info(title)
            if not manga:
                continue

            author_name, artist_name = None, None
            for r in manga["relationships"]:
                if r["type"] == "author":
                    author = self._get_manga_author(r["id"])
                    author_name = author["attributes"]["name"]
                if r["type"] == "artist":
                    artist = self._get_manga_author(r["id"])
                    artist_name = artist["attributes"]["name"]

            chapters = self._get_manga_chapters(manga["id"], lang)
            chapter_count = chapters.get("total", None)
            latest_chapter_published_at = None
            if len(chapters["data"]):
                latest_chapter = chapters["data"][-1]
                latest_chapter_published_at = latest_chapter["attributes"]["publishAt"]

            # Get tags for the selected language
            tags = []
            for tag in manga["attributes"]["tags"]:
                tag_name_dict = tag["attributes"]["name"]
                if lang in tag_name_dict:
                    tags.append(tag_name_dict[lang])

            doc = Document(
                text=manga["attributes"]["title"].get(lang, title),
                extra_info={
                    "id": manga["id"],
                    "author": author_name,
                    "artist": artist_name,
                    "description": manga["attributes"]["description"].get(lang, None),
                    "original_language": manga["attributes"]["originalLanguage"],
                    "tags": tags,
                    "chapter_count": chapter_count,
                    "latest_chapter_published_at": latest_chapter_published_at,
                },
            )
            result.append(doc)

        return result


if __name__ == "__main__":
    reader = MangaDexReader()

    print(reader.load_data(titles=["Grand Blue Dreaming"], lang="en"))
