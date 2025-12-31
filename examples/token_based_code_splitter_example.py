#!/usr/bin/env python3
"""
Example demonstrating the new token-based CodeSplitter functionality.

This example shows how to use both character-based and token-based code splitting
modes to achieve more precise control over chunk sizes when working with language models.
"""

import os
from typing import List

try:
    from llama_index.core.node_parser.text.code import CodeSplitter
    from llama_index.core.schema import Document
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

# Sample Python code for testing
SAMPLE_PYTHON_CODE = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    
    # Initialize the first two Fibonacci numbers
    fib_prev = 0
    fib_curr = 1
    
    # Calculate subsequent Fibonacci numbers
    for i in range(2, n + 1):
        fib_next = fib_prev + fib_curr
        fib_prev = fib_curr
        fib_curr = fib_next
    
    return fib_curr

def factorial(n):
    """Calculate the factorial of n using recursion."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    """A simple calculator class with basic operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history

def main():
    """Main function to demonstrate calculator usage."""
    calc = Calculator()
    
    # Perform some calculations
    sum_result = calc.add(10, 5)
    product_result = calc.multiply(3, 4)
    
    # Calculate Fibonacci and factorial
    fib_10 = fibonacci(10)
    fact_5 = factorial(5)
    
    print(f"Sum: {sum_result}")
    print(f"Product: {product_result}")
    print(f"10th Fibonacci number: {fib_10}")
    print(f"5! = {fact_5}")
    print("History:", calc.get_history())

if __name__ == "__main__":
    main()
'''

def demonstrate_character_based_splitting():
    """Demonstrate traditional character-based code splitting."""
    print("=== Character-based Code Splitting ===")
    
    # Create a character-based splitter
    char_splitter = CodeSplitter(
        language="python",
        count_mode="char",
        max_chars=200,  # Small character limit for demonstration
        chunk_lines=10,
        chunk_lines_overlap=2
    )
    
    chunks = char_splitter.split_text(SAMPLE_PYTHON_CODE)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        char_count = len(chunk)
        print(f"\nChunk {i+1} ({char_count} characters):")
        print("-" * 40)
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


def demonstrate_token_based_splitting():
    """Demonstrate new token-based code splitting."""
    print("\n\n=== Token-based Code Splitting ===")
    
    # Create a token-based splitter
    token_splitter = CodeSplitter(
        language="python",
        count_mode="token",
        max_tokens=50,  # Small token limit for demonstration
        chunk_lines=10,
        chunk_lines_overlap=2
    )
    
    chunks = token_splitter.split_text(SAMPLE_PYTHON_CODE)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        # Get token count using the same tokenizer
        token_count = len(token_splitter._tokenizer(chunk))
        char_count = len(chunk)
        print(f"\nChunk {i+1} ({token_count} tokens, {char_count} characters):")
        print("-" * 50)
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


def demonstrate_custom_tokenizer():
    """Demonstrate using a custom tokenizer."""
    print("\n\n=== Custom Tokenizer Example ===")
    
    def simple_word_tokenizer(text: str) -> List[str]:
        """Simple tokenizer that splits on whitespace and punctuation."""
        import re
        return re.findall(r'\b\w+\b', text)
    
    # Create a splitter with custom tokenizer
    custom_splitter = CodeSplitter(
        language="python",
        count_mode="token",
        max_tokens=30,  # Token limit using custom tokenizer
        tokenizer=simple_word_tokenizer
    )
    
    chunks = custom_splitter.split_text(SAMPLE_PYTHON_CODE)
    
    print(f"Number of chunks with custom tokenizer: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        token_count = len(simple_word_tokenizer(chunk))
        print(f"\nChunk {i+1} ({token_count} word tokens):")
        print("-" * 40)
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


def demonstrate_document_processing():
    """Demonstrate processing documents with token-based splitting."""
    print("\n\n=== Document Processing Example ===")
    
    # Create a document
    document = Document(
        text=SAMPLE_PYTHON_CODE,
        metadata={"file_name": "calculator.py", "language": "python"}
    )
    
    # Create token-based splitter
    splitter = CodeSplitter(
        language="python",
        count_mode="token",
        max_tokens=40,
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    # Process the document into nodes
    nodes = splitter.get_nodes_from_documents([document])
    
    print(f"Created {len(nodes)} nodes from document")
    for i, node in enumerate(nodes):
        token_count = len(splitter._tokenizer(node.text))
        print(f"\nNode {i+1} ({token_count} tokens):")
        print(f"Metadata: {node.metadata}")
        print(f"Text preview: {node.text[:100]}...")


def compare_modes():
    """Compare character vs token-based splitting side by side."""
    print("\n\n=== Character vs Token Mode Comparison ===")
    
    # Test code snippet
    test_code = '''
def complex_function_with_very_long_name():
    variable_with_extremely_long_descriptive_name = "some_string_value"
    another_variable_name = variable_with_extremely_long_descriptive_name.upper()
    return another_variable_name
'''
    
    # Character-based splitting
    char_splitter = CodeSplitter(
        language="python",
        count_mode="char", 
        max_chars=60
    )
    char_chunks = char_splitter.split_text(test_code)
    
    # Token-based splitting
    token_splitter = CodeSplitter(
        language="python",
        count_mode="token",
        max_tokens=15
    )
    token_chunks = token_splitter.split_text(test_code)
    
    print("Character-based chunks:")
    for i, chunk in enumerate(char_chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk.strip()[:50]}...")
    
    print(f"\nToken-based chunks:")
    for i, chunk in enumerate(token_chunks):
        token_count = len(token_splitter._tokenizer(chunk))
        print(f"  Chunk {i+1} ({token_count} tokens): {chunk.strip()[:50]}...")


def main():
    """Run all demonstrations."""
    if not HAS_DEPENDENCIES:
        print("Error: Missing dependencies")
        print("Please install tree_sitter_language_pack and llama_index packages")
        return
        
    try:
        print("Token-based CodeSplitter Examples")
        print("=" * 50)
        
        demonstrate_character_based_splitting()
        demonstrate_token_based_splitting()
        demonstrate_custom_tokenizer()
        demonstrate_document_processing()
        compare_modes()
        
        print("\n\n=== Summary ===")
        print("✅ Character-based splitting: Uses max_chars parameter")
        print("✅ Token-based splitting: Uses max_tokens parameter")
        print("✅ Custom tokenizers: Supported via tokenizer parameter")
        print("✅ Backwards compatible: Existing code continues to work")
        print("✅ Document processing: Full node creation support")
        
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()