#!/usr/bin/env python3
"""
Example usage of the Heroku LLM integration.

This example demonstrates how to use the Heroku Managed Inference LLM
with LlamaIndex for chat and completion tasks.
"""

import os
from llama_index.llms.heroku import Heroku
from llama_index.core.llms import ChatMessage, MessageRole


def setup_environment():
    """Set up environment variables for demonstration."""
    if (
        not os.getenv("INFERENCE_KEY")
        or not os.getenv("INFERENCE_URL")
        or not os.getenv("INFERENCE_MODEL_ID")
    ):
        print(
            "Environment variables not set. Using placeholder values for demonstration."
        )
        os.environ["INFERENCE_KEY"] = "your-inference-key"
        os.environ["INFERENCE_URL"] = "https://us.inference.heroku.com"
        os.environ["INFERENCE_MODEL_ID"] = "claude-3-5-haiku"
    else:
        print("Using existing environment variables.")


def example_1_parameters():
    """Example 1: Using parameters directly."""
    print("\n=== Example 1: Using Parameters ===")

    try:
        # Initialize with parameters
        llm = Heroku(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            is_chat_model=True,
            max_tokens=100,
            temperature=0.5,
        )

        # Chat completion
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, content="You are a helpful assistant."
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="What are the most popular house pets in North America?",
            ),
        ]

        response = llm.chat(messages)
        print(f"Chat Response: {response.message.content}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide valid parameters.")
    except Exception as e:
        print(f"API Error: {e}")
        print("This is expected when using placeholder values.")
        print("To test with real Heroku inference, set valid environment variables.")


def example_2_environment_variables():
    """Example 2: Using environment variables."""
    print("=== Example 2: Using Environment Variables ===")

    # Initialize the LLM using environment variables
    llm = Heroku()

    # Simple text completion
    response = llm.complete(
        "Explain the importance of open source LLMs in one sentence."
    )
    print(f"Completion: {response.text}")


def example_3_text_completion():
    """Example 3: Text completion with non-chat model."""
    print("\n=== Example 3: Text Completion ===")

    llm = Heroku()

    # Text completion
    response = llm.complete("The future of artificial intelligence is")
    print(f"Text Completion: {response.text}")


def main():
    """Main function that runs all examples."""
    # Set up environment variables for demonstration
    setup_environment()

    # Run all examples
    example_1_parameters()
    example_2_environment_variables()
    example_3_text_completion()


if __name__ == "__main__":
    main()
