"""Genius Reader."""

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class GeniusReader(BaseReader):
    """GeniusReader for various operations with lyricsgenius."""

    def __init__(self, access_token: str):
        """Initialize the GeniusReader with an access token."""
        try:
            import lyricsgenius
        except ImportError:
            raise ImportError(
                "Please install lyricsgenius via 'pip install lyricsgenius'"
            )
        self.genius = lyricsgenius.Genius(access_token)

    def load_artist_songs(
        self, artist_name: str, max_songs: Optional[int] = None
    ) -> List[Document]:
        """Load all or a specified number of songs by an artist."""
        artist = self.genius.search_artist(artist_name, max_songs=max_songs)
        return [Document(text=song.lyrics) for song in artist.songs] if artist else []

    def load_all_artist_songs(self, artist_name: str) -> List[Document]:
        artist = self.genius.search_artist(artist_name)
        artist.save_lyrics()
        return [Document(text=song.lyrics) for song in artist.songs]

    def load_artist_songs_with_filters(
        self,
        artist_name: str,
        most_popular: bool = True,
        max_songs: Optional[int] = None,
        max_pages: int = 50,
    ) -> Document:
        """
        Load the most or least popular song of an artist.

        Args:
            artist_name (str): The artist's name.
            most_popular (bool): True for most popular, False for least popular song.
            max_songs (Optional[int]): Maximum number of songs to consider for popularity.
            max_pages (int): Maximum number of pages to fetch.

        Returns:
            Document: A document containing lyrics of the most/least popular song.

        """
        artist = self.genius.search_artist(artist_name, max_songs=1)
        if not artist:
            return None

        songs_fetched = 0
        page = 1
        songs = []
        while (
            page
            and page <= max_pages
            and (max_songs is None or songs_fetched < max_songs)
        ):
            request = self.genius.artist_songs(
                artist.id, sort="popularity", per_page=50, page=page
            )
            songs.extend(request["songs"])
            songs_fetched += len(request["songs"])
            page = (
                request["next_page"]
                if (max_songs is None or songs_fetched < max_songs)
                else None
            )

        target_song = songs[0] if most_popular else songs[-1]
        song_details = self.genius.search_song(target_song["title"], artist.name)
        return Document(text=song_details.lyrics) if song_details else None

    def load_song_by_url_or_id(
        self, song_url: Optional[str] = None, song_id: Optional[int] = None
    ) -> List[Document]:
        """Load song by URL or ID."""
        if song_url:
            song = self.genius.song(url=song_url)
        elif song_id:
            song = self.genius.song(song_id)
        else:
            return []

        return [Document(text=song.lyrics)] if song else []

    def search_songs_by_lyrics(self, lyrics: str) -> List[Document]:
        """
        Search for songs by a snippet of lyrics.

        Args:
            lyrics (str): The lyric snippet you're looking for.

        Returns:
            List[Document]: A list of documents containing songs with those lyrics.

        """
        search_results = self.genius.search_songs(lyrics)
        songs = search_results["hits"] if search_results else []

        results = []
        for hit in songs:
            song_url = hit["result"]["url"]
            song_lyrics = self.genius.lyrics(song_url=song_url)
            results.append(Document(text=song_lyrics))

        return results

    def load_songs_by_tag(
        self, tag: str, max_songs: Optional[int] = None, max_pages: int = 50
    ) -> List[Document]:
        """
        Load songs by a specific tag.

        Args:
            tag (str): The tag or genre to load songs for.
            max_songs (Optional[int]): Maximum number of songs to fetch. If None, no specific limit.
            max_pages (int): Maximum number of pages to fetch.

        Returns:
            List[Document]: A list of documents containing song lyrics.

        """
        lyrics = []
        total_songs_fetched = 0
        page = 1

        while (
            page
            and page <= max_pages
            and (max_songs is None or total_songs_fetched < max_songs)
        ):
            res = self.genius.tag(tag, page=page)
            for hit in res["hits"]:
                if max_songs is None or total_songs_fetched < max_songs:
                    song_lyrics = self.genius.lyrics(song_url=hit["url"])
                    lyrics.append(Document(text=song_lyrics))
                    total_songs_fetched += 1
                else:
                    break
            page = (
                res["next_page"]
                if max_songs is None or total_songs_fetched < max_songs
                else None
            )

        return lyrics


if __name__ == "__main__":
    access_token = ""
    reader = GeniusReader(access_token)
    # Example usage
    print(reader.load_artist_songs("Chance the Rapper", max_songs=1))
