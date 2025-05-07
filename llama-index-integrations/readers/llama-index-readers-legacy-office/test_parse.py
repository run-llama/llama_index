"""Test script for LegacyOfficeReader."""

import logging
from llama_index.readers.legacy_office import LegacyOfficeReader
from llama_index.core.readers import SimpleDirectoryReader
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    # Initialize reader
    reader = LegacyOfficeReader(
        excluded_embed_metadata_keys=["file_path", "file_name"],
        excluded_llm_metadata_keys=["file_type"],
    )
    # Test document with LegacyOfficeReader
    test_document(reader, "./test_dir/harry_potter_lagacy.doc")

    # Test document with SimpleDirectoryReader
    test_simple_directory_reader("./test_dir/")


def test_document(reader: LegacyOfficeReader, file_path: str):
    """Test loading and parsing a document."""
    try:
        # Load and parse the document
        documents = reader.load_data(Path(file_path))
        print("\n" + "=" * 80)
        print(f"✨ Successfully parsed document into {len(documents)} parts ✨")
        print("=" * 80)

        # Print document content
        if documents:
            print("\n📄 Document Content")
            print("=" * 80)
            print("First 500 characters:")
            print("-" * 80)
            print(documents[0].text[:500])

            print("\n📋 Document Metadata")
            print("=" * 80)
            print("📌 Basic File Information:")
            print("-" * 40)
            print(f"📂 File Path  : {documents[0].metadata.get('file_path')}")
            print(f"📄 File Name  : {documents[0].metadata.get('file_name')}")
            print(f"🏷️  File Type  : {documents[0].metadata.get('file_type')}")

            print("\n🔍 Tika Metadata:")
            print("-" * 40)
            for key, value in sorted(documents[0].metadata.items()):
                if key not in ["file_path", "file_name", "file_type"]:
                    print(f"• {key.title():20}: {value}")

    except Exception as e:
        print("\n❌ Error parsing document:")
        print("-" * 40)
        print(f"Error: {e!s}")
        print("\nBacktrace:")
        print("-" * 40)
        import traceback

        print(traceback.format_exc())


def test_simple_directory_reader(input_dir: str):
    """Test loading and parsing a document using SimpleDirectoryReader."""
    try:
        # Load and parse the document using SimpleDirectoryReader
        reader = SimpleDirectoryReader(
            input_dir=input_dir, file_extractor={".doc": LegacyOfficeReader()}
        )
        documents = reader.load_data()

        print("\n" + "=" * 80)
        print("📚 Testing SimpleDirectoryReader with .doc file")
        print("=" * 80)
        print(f"✨ Successfully parsed document into {len(documents)} parts ✨")

        # Print document content
        if documents:
            print("\n📄 Document Content")
            print("=" * 80)
            print("First 500 characters:")
            print("-" * 80)
            print(documents[0].text[:500])

    except Exception as e:
        print("\n❌ Error parsing document with SimpleDirectoryReader:")
        print("-" * 40)
        print(f"Error: {e!s}")
        print("\nBacktrace:")
        print("-" * 40)
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
