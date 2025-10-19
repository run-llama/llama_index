"""
Example: Using Custom SVG Parser with Confluence Reader

This example demonstrates how to use a custom parser for SVG files if you want
to handle SVG processing differently or avoid the pycairo dependency issues.

Option 1: Skip SVG processing entirely (default behavior without svg extra)
Option 2: Use the built-in SVG processor (requires pip install llama-index-readers-confluence[svg])
Option 3: Provide a custom SVG parser (example below)
"""

from typing import List, Union
import pathlib
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import FileType


# Example 1: Simple custom SVG parser that extracts text content from SVG
class SimpleSVGParser(BaseReader):
    """
    Simple SVG parser that extracts text elements from SVG files.
    This avoids the pycairo dependency by using basic XML parsing.
    """

    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        """Load and parse an SVG file to extract text content."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree.ElementTree is required")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            # Parse SVG XML
            root = ET.fromstring(content)
            # Extract all text elements (handles common SVG namespace)
            ns = {"svg": "http://www.w3.org/2000/svg"}
            texts = []

            # Try with namespace
            for text_elem in root.findall(".//svg:text", ns):
                if text_elem.text:
                    texts.append(text_elem.text.strip())

            # Try without namespace if nothing found
            if not texts:
                for text_elem in root.findall(".//text"):
                    if text_elem.text:
                        texts.append(text_elem.text.strip())

            extracted_text = " ".join(texts) if texts else "[SVG Image - No text content]"

            return [
                Document(
                    text=extracted_text,
                    metadata={"file_path": str(file_path), "source_type": "svg"},
                )
            ]
        except Exception as e:
            return [
                Document(
                    text=f"[Error parsing SVG: {str(e)}]",
                    metadata={"file_path": str(file_path), "source_type": "svg"},
                )
            ]


# Example 2: Custom SVG parser using cairosvg (alternative to svglib)
class CairoSVGParser(BaseReader):
    """
    Alternative SVG parser using cairosvg library.
    Install with: pip install cairosvg pillow pytesseract

    Note: This still requires cairo system libraries but has different
    installation characteristics than svglib+pycairo.
    """

    def load_data(
        self, file_path: Union[str, pathlib.Path], **kwargs
    ) -> List[Document]:
        """Load and parse an SVG file by converting to PNG and extracting text."""
        try:
            import cairosvg
            import pytesseract
            from PIL import Image
            from io import BytesIO
        except ImportError:
            raise ImportError(
                "cairosvg, pillow, and pytesseract are required. "
                "Install with: pip install cairosvg pillow pytesseract"
            )

        try:
            # Convert SVG to PNG
            png_data = cairosvg.svg2png(url=str(file_path))

            # Extract text using OCR
            image = Image.open(BytesIO(png_data))
            text = pytesseract.image_to_string(image)

            return [
                Document(
                    text=text or "[SVG Image - No text extracted]",
                    metadata={"file_path": str(file_path), "source_type": "svg"},
                )
            ]
        except Exception as e:
            return [
                Document(
                    text=f"[Error parsing SVG: {str(e)}]",
                    metadata={"file_path": str(file_path), "source_type": "svg"},
                )
            ]


# Usage examples

def example_without_svg_support():
    """
    Example 1: Use Confluence reader without SVG support.
    SVG attachments will be skipped with a warning.
    """
    reader = ConfluenceReader(
        base_url="https://yoursite.atlassian.com/wiki",
        api_token="your_token",
    )

    # SVG attachments will be skipped automatically
    documents = reader.load_data(
        space_key="MYSPACE",
        include_attachments=True,
    )
    return documents


def example_with_builtin_svg_support():
    """
    Example 2: Use built-in SVG support.
    Requires: pip install llama-index-readers-confluence[svg]
    """
    reader = ConfluenceReader(
        base_url="https://yoursite.atlassian.com/wiki",
        api_token="your_token",
    )

    # Built-in SVG processing will be used if dependencies are installed
    documents = reader.load_data(
        space_key="MYSPACE",
        include_attachments=True,
    )
    return documents


def example_with_custom_svg_parser():
    """
    Example 3: Use custom SVG parser to avoid pycairo dependency.
    This gives you full control over SVG processing.
    """
    # Use the simple text extraction parser
    svg_parser = SimpleSVGParser()

    reader = ConfluenceReader(
        base_url="https://yoursite.atlassian.com/wiki",
        api_token="your_token",
        custom_parsers={
            FileType.SVG: svg_parser,
        },
    )

    documents = reader.load_data(
        space_key="MYSPACE",
        include_attachments=True,
    )
    return documents


def example_skip_svg_via_callback():
    """
    Example 4: Skip SVG attachments using a callback.
    This is useful if you want to explicitly skip SVG files.
    """
    def attachment_filter(media_type: str, file_size: int, title: str) -> tuple[bool, str]:
        # Skip SVG files
        if media_type == "image/svg+xml":
            return False, "SVG files are not supported in this configuration"
        return True, ""

    reader = ConfluenceReader(
        base_url="https://yoursite.atlassian.com/wiki",
        api_token="your_token",
        process_attachment_callback=attachment_filter,
    )

    documents = reader.load_data(
        space_key="MYSPACE",
        include_attachments=True,
    )
    return documents


if __name__ == "__main__":
    print("SVG Processing Examples for Confluence Reader")
    print("=" * 50)
    print("\nOption 1: Without SVG support (default)")
    print("  - No additional dependencies required")
    print("  - SVG attachments are skipped with warnings")
    print("  - Best for systems where pycairo cannot be installed")

    print("\nOption 2: With built-in SVG support")
    print("  - Requires: pip install llama-index-readers-confluence[svg]")
    print("  - Full OCR-based text extraction from SVG")
    print("  - May have installation challenges on some systems")

    print("\nOption 3: With custom SVG parser")
    print("  - No pycairo dependency")
    print("  - Simple text element extraction")
    print("  - Easy to customize for your needs")

    print("\nOption 4: Skip SVG via callback")
    print("  - Explicitly filter out SVG files")
    print("  - Clean logs without warnings")
    print("  - Useful when SVG content is not needed")
