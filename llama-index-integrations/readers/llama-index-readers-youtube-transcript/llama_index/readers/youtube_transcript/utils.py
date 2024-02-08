import re

# regular expressions to match the different syntax of YouTube links
YOUTUBE_URL_PATTERNS = [
    r"^https?://(?:www\.)?youtube\.com/watch\?v=([\w-]+)",
    r"^https?://(?:www\.)?youtube\.com/embed/([\w-]+)",
    r"^https?://youtu\.be/([\w-]+)",  # youtu.be does not use www
]


def is_youtube_video(url: str) -> bool:
    """
    Returns whether the passed in `url` matches the various YouTube URL formats.
    """
    for pattern in YOUTUBE_URL_PATTERNS:
        if re.search(pattern, url):
            return True
    return False
