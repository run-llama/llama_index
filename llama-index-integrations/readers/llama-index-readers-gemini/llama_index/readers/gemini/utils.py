"""Utility functions for Gemini PDF Reader."""

import logging
import os
import tempfile
import requests
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)


def is_web_url(file_path: str) -> bool:
    """
    Check if the input is a web URL.

    Args:
        file_path: Path to check

    Returns:
        True if the input is a web URL, False otherwise
    """
    try:
        parsed = urlparse(file_path)
        return bool(parsed.scheme in ["http", "https"] and parsed.netloc)
    except Exception:
        return False


def download_from_url(url: str, verbose: bool = False) -> str:
    """
    Download a PDF from a URL.

    Args:
        url: URL to download from
        verbose: Whether to print verbose messages

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If download fails
    """
    if verbose:
        logger.info(f"Downloading PDF from URL: {url}")

    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"gemini_pdf_download_{hash(url)}.pdf")

        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
            if verbose:
                logger.warning(f"URL may not be a PDF (Content-Type: {content_type})")

        # Save the file
        with open(temp_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if verbose:
            logger.info(f"Downloaded PDF to temporary file: {temp_file_path}")

        return temp_file_path

    except Exception as e:
        if verbose:
            logger.error(f"Error downloading from URL {url}: {e!s}")
        raise RuntimeError(f"Failed to download file: {e!s}") from e
