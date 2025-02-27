#!/usr/bin/env python3
"""
Direct test script for FireCrawl Web Reader integration.
"""

import os
import sys
from dotenv import load_dotenv

# Add the FireCrawl module path to the Python path
sys.path.append("llama-index-integrations/readers/llama-index-readers-web/llama_index/readers/web/firecrawl_web")

# Load environment variables from .env file (if it exists)
load_dotenv()

# Get the API key from environment variables
api_key = os.environ.get("FIRECRAWL_API_KEY")

if not api_key:
    print("Error: FIRECRAWL_API_KEY environment variable not set.")
    print("Please set it by running: export FIRECRAWL_API_KEY=your_api_key")
    print("Or update the .env file with your API key.")
    sys.exit(1)

# Import the FireCrawlWebReader directly
try:
    from base import FireCrawlWebReader
    print("Successfully imported FireCrawlWebReader")
except ImportError as e:
    print(f"Error importing FireCrawlWebReader: {e}")
    sys.exit(1)

def test_scrape_mode():
    """Test the FireCrawlWebReader in scrape mode."""
    print("\n=== Testing FireCrawlWebReader in scrape mode ===")
    
    try:
        # Initialize the FireCrawlWebReader with your API key and scrape mode
        firecrawl_reader = FireCrawlWebReader(
            api_key=api_key,
            mode="scrape",
        )
        print("Successfully initialized FireCrawlWebReader")
        
        # URL to scrape
        url = "https://mairistumpf.com/"
        
        # Load documents from the URL
        print(f"Loading data from URL: {url}")
        documents = firecrawl_reader.load_data(url=url)
        
        # Print the number of documents loaded
        print(f"Successfully loaded {len(documents)} document(s)")
        
        # Print the first few characters of each document
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1} preview:")
            print(f"Text length: {len(doc.text)} characters")
            print(f"Text preview: {doc.text[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        return documents
    except Exception as e:
        print(f"Error in scrape mode: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_crawl_mode():
    """Test the FireCrawlWebReader in crawl mode."""
    print("\n=== Testing FireCrawlWebReader in crawl mode ===")
    
    try:
        # Initialize the FireCrawlWebReader with your API key and crawl mode
        firecrawl_reader = FireCrawlWebReader(
            api_key=api_key,
            mode="crawl",
            params={"maxDepth": 2, "excludePaths": ["articles"]}
        )
        print("Successfully initialized FireCrawlWebReader in crawl mode")
        
        # URL to crawl
        url = "https://www.paulgraham.com"
        
        # Load documents from the URL
        print(f"Crawling URL: {url}")
        documents = firecrawl_reader.load_data(url=url)
        
        # Print the number of documents loaded
        print(f"Successfully loaded {len(documents)} document(s)")
        
        # Print the first few characters of each document
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1} preview:")
            print(f"Text length: {len(doc.text)} characters")
            print(f"Text preview: {doc.text[:100]}...")
            print(f"URL: {doc.metadata.get('url', 'N/A')}")
        
        return documents
    except Exception as e:
        print(f"Error in crawl mode: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def test_search_mode():
    """Test the FireCrawlWebReader in search mode."""
    print("\n=== Testing FireCrawlWebReader in search mode ===")
    
    try:
        # Initialize the FireCrawlWebReader with your API key and search mode
        firecrawl_reader = FireCrawlWebReader(
            api_key=api_key,
            mode="search",
            params={
                "limit": 5,
                "lang": "en",
                "country": "us"
            }
        )   
        print("Successfully initialized FireCrawlWebReader in search mode")

        # Query to search
        query = "firecrawl"
        
        # Load documents using the query parameter
        print(f"Searching for query: {query}")
        documents = firecrawl_reader.load_data(query=query)
        
        # Print the number of documents loaded
        print(f"Successfully loaded {len(documents)} document(s)")
        
        # Print the first few characters of each document
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1} preview:")
            print(f"Text length: {len(doc.text)} characters")
            print(f"Text preview: {doc.text[:100]}...")
            print(f"URL: {doc.metadata.get('url', 'N/A')}")
        
        return documents
    except Exception as e:
        print(f"Error in search mode: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_extract_mode():
    """Test the FireCrawlWebReader in extract mode."""
    print("\n=== Testing FireCrawlWebReader in extract mode ===")
    
    try:
        # Initialize the FireCrawlWebReader with your API key and extract mode
        firecrawl_reader = FireCrawlWebReader(
            api_key=api_key,
            mode="extract",
            params={
                "prompt": "How many years of experience does the author have?", 
                "showSources": True,
                "includeSubdomains": True
            }
        )
        print("Successfully initialized FireCrawlWebReader in extract mode")

        # URLs to extract
        urls = ["https://mairistumpf.com/*"]
        
        # Load documents from the URLs
        print(f"Extracting data from URLs: {urls}")
        
        # Add a debug hook to see the raw response
        original_extract = firecrawl_reader.firecrawl.extract
        
        def debug_extract(*args, **kwargs):
            response = original_extract(*args, **kwargs)
            print(f"\nDEBUG - Raw extract response: {response}")
            return response
        
        firecrawl_reader.firecrawl.extract = debug_extract
        
        documents = firecrawl_reader.load_data(urls=urls)
        
        # Print the number of documents loaded
        print(f"Successfully loaded {len(documents)} document(s)")
        
        # Print the first few characters of each document
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1} preview:")
            print(f"Text length: {len(doc.text)} characters")
            print(f"Text preview: {doc.text[:100]}...")
            print(f"Metadata: {doc.metadata}")
        
        return documents
    except Exception as e:
        print(f"Error in extract mode: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the tests."""
    print("Starting FireCrawl Web Reader direct test...")
    
    # Test scrape mode
    test_scrape_mode()
    
    # Test crawl mode
    test_crawl_mode()

    # Test search mode
    test_search_mode()

    # Test extract mode
    test_extract_mode()

if __name__ == "__main__":
    main() 