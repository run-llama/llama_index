import re
import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class BookNotFoundException(Exception):
    pass


class ProjectGutenbergReader(BaseReader):
    def load_data(self, book_title):
        document = maybe_get_gutenberg_book(book_title)
        return [document]


def maybe_get_gutenberg_book(title):
    url = f"http://gutendex.com/books/?search={title}"
    response = requests.get(url)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise BookNotFoundException(f"Search request failed for title: {title}")

    books = response.json().get("results", [])
    book_id = None
    for book in books:
        if title.lower() in book["title"].lower():
            book_id = book["id"]
            break

    if book_id is None:
        raise BookNotFoundException(f"Book not found with title: {title}")

    book_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    response = requests.get(book_url)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise BookNotFoundException(f"Book not found with title: {title}")

    text = response.text
    # Get rid of binary characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    return Document(text=text)
