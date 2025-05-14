"""Spotify reader."""

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SpotifyReader(BaseReader):
    """
    Spotify Reader.

    Read a user's saved albums, tracks, or playlists from Spotify.

    """

    def load_data(self, collection: Optional[str] = "albums") -> List[Document]:
        """
        Load data from a user's Spotify account.

        Args:
            collections (Optional[str]): "albums", "tracks", or "playlists"

        """
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        scope = "user-library-read"
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

        results = []

        if collection == "albums":
            response = sp.current_user_saved_albums()
            items = response["items"]
            for item in items:
                album = item["album"]
                album_name = album["name"]
                artist_name = album["artists"][0]["name"]
                album_string = f"Album {album_name} by Artist {artist_name}\n"
                results.append(Document(text=album_string))
        elif collection == "tracks":
            response = sp.current_user_saved_tracks()
            items = response["items"]
            for item in items:
                track = item["track"]
                track_name = track["name"]
                artist_name = track["artists"][0]["name"]
                artist_string = f"Track {track_name} by Artist {artist_name}\n"
                results.append(Document(text=artist_string))
        elif collection == "playlists":
            response = sp.current_user_playlists()
            items = response["items"]
            for item in items:
                playlist_name = item["name"]
                owner_name = item["owner"]["display_name"]
                playlist_string = f"Playlist {playlist_name} created by {owner_name}\n"
                results.append(Document(text=playlist_string))
        else:
            raise ValueError(
                "Invalid collection parameter value. Allowed values are 'albums',"
                " 'tracks', or 'playlists'."
            )

        return results


if __name__ == "__main__":
    reader = SpotifyReader()
    print(reader.load_data())
