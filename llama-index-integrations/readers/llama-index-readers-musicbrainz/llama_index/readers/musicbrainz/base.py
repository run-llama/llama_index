"""MusicBrainz reader."""

from typing import List, Literal, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

DEFAULT_USER_AGENT = "llama-index-readers-musicbrainz/0.1.0 ( https://github.com/run-llama/llama_index )"

EntityType = Literal["artist", "release-group", "release", "recording", "work", "label"]


class MusicBrainzReader(BaseReader):
    """
    MusicBrainz Reader.

    Read structured music metadata from the open MusicBrainz database
    (https://musicbrainz.org). Supports two access patterns:

    - **Search** an entity type by free-text query and load each hit as a
      ``Document`` with the entity's human-readable summary as text and
      its full JSON payload (MBID, scores, relationships, tags) preserved
      in ``metadata``.
    - **Lookup** a single entity by its MusicBrainz Identifier (MBID),
      optionally requesting relationships, tags, and ratings via the
      ``includes`` parameter.

    The MusicBrainz Web Service is free, requires no API key, and is
    rate-limited to one request per second per User-Agent. A polite
    User-Agent string identifying your application is mandatory per the
    MusicBrainz API guidelines; supply one via ``user_agent`` or rely on
    the library default (which identifies as ``llama-index-readers-musicbrainz``).

    See https://musicbrainz.org/doc/MusicBrainz_API for the canonical API
    reference.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        host: Optional[str] = None,
    ) -> None:
        """
        Initialize the reader.

        Args:
            user_agent: Identifying string sent with every request. The
                MusicBrainz API rejects requests without a meaningful
                User-Agent; the default identifies this reader, but
                downstream applications should pass an app-specific
                string (e.g. ``"my-app/1.0 (contact@example.com)"``).
            host: Optional alternative host (e.g. a local mirror). When
                ``None``, the public ``musicbrainz.org`` host is used.
        """
        import musicbrainzngs

        self._client = musicbrainzngs
        # set_useragent accepts (app, version, contact); we pre-formatted
        # user_agent so we pass it as the full app string and leave the
        # other fields blank for the library to handle.
        self._client.set_useragent("llama-index", "0.1.0", user_agent)
        if host is not None:
            self._client.set_hostname(host)

    def load_data(
        self,
        query: Optional[str] = None,
        entity: EntityType = "artist",
        mbid: Optional[str] = None,
        includes: Optional[List[str]] = None,
        limit: int = 25,
    ) -> List[Document]:
        """
        Load MusicBrainz entities as Documents.

        Args:
            query: Free-text search query (e.g. ``"Radiohead"``). Required
                when ``mbid`` is not supplied. The query is matched
                against the entity's primary fields per MusicBrainz Lucene
                syntax.
            entity: Which entity type to search or look up. One of
                ``"artist"``, ``"release-group"``, ``"release"``,
                ``"recording"``, ``"work"``, ``"label"``.
            mbid: MusicBrainz Identifier. When set, ``query`` is ignored
                and the reader performs a single lookup; the result is
                returned as one Document. ``includes`` may be passed to
                hydrate sub-resources (e.g. ``["release-groups", "tags"]``).
            includes: List of sub-resources to include on lookup requests.
                Ignored for searches. Valid values depend on the entity
                type — see the MusicBrainz API documentation.
            limit: Maximum number of search hits to return (1–100,
                defaults to 25). Ignored for lookup requests.

        Returns:
            A list of ``Document`` objects. Search results return one
            Document per hit; lookups return at most one Document.

        Raises:
            ValueError: If neither ``query`` nor ``mbid`` is supplied,
                or if ``entity`` is not one of the supported types.
        """
        if mbid is None and not query:
            raise ValueError("Either 'query' or 'mbid' must be supplied.")
        if entity not in {
            "artist",
            "release-group",
            "release",
            "recording",
            "work",
            "label",
        }:
            raise ValueError(
                f"Unsupported entity type: {entity!r}. "
                "Use 'artist', 'release-group', 'release', 'recording', "
                "'work', or 'label'."
            )

        if mbid is not None:
            return self._lookup(entity=entity, mbid=mbid, includes=includes or [])
        return self._search(entity=entity, query=query, limit=limit)

    def _search(self, entity: EntityType, query: str, limit: int) -> List[Document]:
        method = getattr(self._client, f"search_{entity.replace('-', '_')}s")
        response = method(query=query, limit=limit)
        results_key = f"{entity}-list"
        hits = response.get(results_key, [])
        documents: List[Document] = []
        for hit in hits:
            text = _format_hit(entity, hit)
            documents.append(Document(text=text, metadata={"entity": entity, **hit}))
        return documents

    def _lookup(
        self, entity: EntityType, mbid: str, includes: List[str]
    ) -> List[Document]:
        method = getattr(self._client, f"get_{entity.replace('-', '_')}_by_id")
        response = method(mbid, includes=includes)
        payload = response.get(entity, {})
        if not payload:
            return []
        text = _format_hit(entity, payload)
        return [Document(text=text, metadata={"entity": entity, **payload})]


def _format_hit(entity: EntityType, payload: dict) -> str:
    """Render an entity as a short human-readable summary suitable for
    embedding. The verbose fields stay in the Document's metadata so the
    summary can keep the embedding signal focused.
    """
    name = payload.get("name") or payload.get("title") or ""
    if entity == "artist":
        country = payload.get("country") or "?"
        return f"Artist {name} ({country})"
    if entity == "release-group":
        artist_credit = payload.get("artist-credit-phrase") or _first_artist(payload)
        primary_type = payload.get("primary-type") or "release group"
        return f"{primary_type} {name} by {artist_credit}"
    if entity == "release":
        artist_credit = payload.get("artist-credit-phrase") or _first_artist(payload)
        date = payload.get("date") or ""
        return f"Release {name} by {artist_credit} ({date})".strip()
    if entity == "recording":
        artist_credit = payload.get("artist-credit-phrase") or _first_artist(payload)
        return f"Recording {name} by {artist_credit}"
    if entity == "work":
        return f"Work {name}"
    if entity == "label":
        country = payload.get("country") or "?"
        return f"Label {name} ({country})"
    return name


def _first_artist(payload: dict) -> str:
    credit = payload.get("artist-credit") or []
    if not credit:
        return "Unknown Artist"
    head = credit[0]
    if isinstance(head, dict):
        artist = head.get("artist") or {}
        return artist.get("name", "Unknown Artist")
    return str(head)
