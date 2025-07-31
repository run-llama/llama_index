#!/usr/bin/env python3
"""
Example usage of the Heroku LLM integration.

This example demonstrates how to use the Heroku Managed Inference LLM
with LlamaIndex for chat and completion tasks.
"""

import os
from llama_index.llms.heroku import Heroku
from llama_index.core.llms import ChatMessage, MessageRole


def main():
    """Main example function."""
    # Example 1: Using environment variables
    print("=== Example 1: Using Environment Variables ===")

    # Check if environment variables are set (in real usage, these would be set in your environment)
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

    try:
        # Initialize the LLM using environment variables
        llm = Heroku()

        # Simple text completion
        response = llm.complete(
            "Explain the importance of open source LLMs in one sentence."
        )
        print(f"Completion: {response.text}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the required environment variables or use parameters.")
    except Exception as e:
        print(f"API Error: {e}")
        print("This is expected when using placeholder values.")
        print("To test with real Heroku inference, set valid environment variables:")
        print("  export INFERENCE_KEY='your-actual-key'")
        print("  export INFERENCE_URL='https://us.inference.heroku.com'")
        print("  export INFERENCE_MODEL_ID='your-model-id'")

    # Example 2: Using parameters directly
    print("\n=== Example 2: Using Parameters ===")

    try:
        # Initialize with parameters
        llm = Heroku(
            model=os.getenv("INFERENCE_MODEL_ID", "claude-3-5-haiku"),
            api_key=os.getenv("INFERENCE_KEY", "your-api-key"),
            inference_url=os.getenv("INFERENCE_URL", "https://us.inference.heroku.com"),
            is_chat_model=True,
            max_tokens=100,
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

    # Example 3: Text completion with non-chat model
    print("\n=== Example 3: Text Completion ===")

    try:
        # Initialize for text completion
        llm = Heroku(
            model=os.getenv("INFERENCE_MODEL_ID", "claude-3-5-haiku"),
            api_key=os.getenv("INFERENCE_KEY", "your-api-key"),
            inference_url=os.getenv("INFERENCE_URL", "https://us.inference.heroku.com"),
            is_chat_model=True,
            max_tokens=50,
        )

        # Text completion
        response = llm.complete("The future of artificial intelligence is")
        print(f"Text Completion: {response.text}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide valid parameters.")
    except Exception as e:  # Catch-all for other potential API errors
        print(f"API Error: {e}")
        print("This is expected when using placeholder values.")
        print("To test with real Heroku inference, set valid environment variables.")


if __name__ == "__main__":
    main()
