"""
LlamaIndex integration example for 0G Compute Network LLM.

This example demonstrates how to use ZeroGLLM with LlamaIndex
components like query engines and chat engines.
"""

import os
import tempfile
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.zerog import ZeroGLLM


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = [
        Document(
            text="""
            The 0G Network is a decentralized AI infrastructure that provides scalable, 
            secure, and cost-effective solutions for AI applications. It consists of 
            three main components: 0G Chain (blockchain layer), 0G Storage (decentralized 
            storage), and 0G Compute (AI inference network).
            
            The 0G Compute Network enables developers to access GPU resources from 
            distributed providers, offering competitive pricing and verification 
            capabilities through Trusted Execution Environments (TEE).
            """,
            metadata={"source": "0g_overview.txt", "topic": "0G Network Overview"}
        ),
        Document(
            text="""
            Decentralized AI offers several advantages over traditional centralized 
            approaches: improved privacy through distributed processing, reduced 
            single points of failure, competitive pricing through market dynamics, 
            and enhanced transparency through blockchain-based verification.
            
            The 0G Network implements these principles by providing a marketplace 
            where GPU providers can offer their compute resources while maintaining 
            cryptographic proof of computation integrity.
            """,
            metadata={"source": "decentralized_ai.txt", "topic": "Decentralized AI Benefits"}
        ),
        Document(
            text="""
            Setting up the 0G Compute Network requires an Ethereum wallet with OG tokens 
            for payment. Developers can choose from official models like llama-3.3-70b-instruct 
            and deepseek-r1-70b, or connect to custom providers.
            
            The network supports standard OpenAI-compatible APIs, making integration 
            straightforward for existing applications. Verification is handled 
            automatically through TEE technology.
            """,
            metadata={"source": "setup_guide.txt", "topic": "Setup and Configuration"}
        )
    ]
    
    return documents


def query_engine_example():
    """Demonstrate using ZeroGLLM with a query engine."""
    print("=== Query Engine Example ===")
    
    # Initialize the 0G LLM
    llm = ZeroGLLM(
        model="llama-3.3-70b-instruct",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        temperature=0.1,  # Low temperature for factual queries
        max_tokens=512
    )
    
    # Set as the default LLM for LlamaIndex
    Settings.llm = llm
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create index
    print("Creating vector index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine
    query_engine = index.as_query_engine(
        response_mode="compact",
        verbose=True
    )
    
    # Ask questions
    questions = [
        "What is the 0G Network?",
        "What are the benefits of decentralized AI?",
        "How do I set up the 0G Compute Network?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = query_engine.query(question)
        print(f"Answer: {response.response}")
        
        # Show source information
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("Sources:")
            for i, node in enumerate(response.source_nodes):
                metadata = node.node.metadata
                print(f"  {i+1}. {metadata.get('source', 'Unknown')} - {metadata.get('topic', 'N/A')}")
    
    print()


def chat_engine_example():
    """Demonstrate using ZeroGLLM with a chat engine."""
    print("=== Chat Engine Example ===")
    
    # Initialize with the reasoning model for better conversation
    llm = ZeroGLLM(
        model="deepseek-r1-70b",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        temperature=0.3,
        max_tokens=1024
    )
    
    Settings.llm = llm
    
    # Create documents and index
    documents = create_sample_documents()
    index = VectorStoreIndex.from_documents(documents)
    
    # Create chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose=True
    )
    
    # Simulate a conversation
    conversation = [
        "Hi! Can you tell me about the 0G Network?",
        "What makes it different from traditional cloud computing?",
        "How does the verification system work?",
        "What do I need to get started?"
    ]
    
    print("Starting conversation with 0G-powered chat engine:")
    print("-" * 50)
    
    for user_message in conversation:
        print(f"User: {user_message}")
        response = chat_engine.chat(user_message)
        print(f"Assistant: {response.response}")
        print()
    
    # Show chat history
    print("Chat History:")
    for i, message in enumerate(chat_engine.chat_history):
        role = "User" if message.role == MessageRole.USER else "Assistant"
        print(f"{i+1}. {role}: {message.content[:100]}...")
    
    print()


def custom_prompt_example():
    """Demonstrate custom prompting with 0G LLM."""
    print("=== Custom Prompt Example ===")
    
    llm = ZeroGLLM(
        model="llama-3.3-70b-instruct",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        temperature=0.5
    )
    
    # Create documents
    documents = create_sample_documents()
    index = VectorStoreIndex.from_documents(documents)
    
    # Custom query engine with specific prompt
    from llama_index.core import PromptTemplate
    
    custom_prompt = PromptTemplate(
        """
        You are an expert on decentralized AI and blockchain technology, specifically 
        the 0G Network. Use the provided context to answer questions accurately and 
        provide practical guidance.
        
        Context information:
        {context_str}
        
        Question: {query_str}
        
        Please provide a comprehensive answer that includes:
        1. Direct answer to the question
        2. Technical details when relevant
        3. Practical implications or next steps
        
        Answer:
        """
    )
    
    query_engine = index.as_query_engine(
        text_qa_template=custom_prompt,
        response_mode="tree_summarize"
    )
    
    question = "How can I integrate 0G Compute Network into my existing AI application?"
    print(f"Question: {question}")
    
    response = query_engine.query(question)
    print(f"Custom-prompted response: {response.response}")
    print()


def streaming_query_example():
    """Demonstrate streaming responses with query engine."""
    print("=== Streaming Query Example ===")
    
    llm = ZeroGLLM(
        model="llama-3.3-70b-instruct",
        private_key=os.getenv("ETHEREUM_PRIVATE_KEY", "your_private_key_here"),
        temperature=0.2
    )
    
    Settings.llm = llm
    
    # Create documents and index
    documents = create_sample_documents()
    index = VectorStoreIndex.from_documents(documents)
    
    # Create streaming query engine
    query_engine = index.as_query_engine(
        streaming=True,
        response_mode="compact"
    )
    
    question = "Explain the architecture and benefits of the 0G Network in detail."
    print(f"Question: {question}")
    print("Streaming response:")
    
    # Stream the response
    streaming_response = query_engine.query(question)
    for chunk in streaming_response.response_gen:
        print(chunk, end="", flush=True)
    
    print("\n")


def main():
    """Run all integration examples."""
    print("0G Compute Network + LlamaIndex Integration Examples")
    print("=" * 60)
    print()
    
    # Check if private key is set
    if not os.getenv("ETHEREUM_PRIVATE_KEY"):
        print("Warning: ETHEREUM_PRIVATE_KEY environment variable not set.")
        print("Using placeholder value for demonstration.")
        print("Set your actual private key as an environment variable for real usage.")
        print()
    
    try:
        # Run examples
        query_engine_example()
        chat_engine_example()
        custom_prompt_example()
        streaming_query_example()
        
        print("All integration examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("- pip install llama-index-core")
        print("- pip install llama-index-llms-0g")


if __name__ == "__main__":
    main()
