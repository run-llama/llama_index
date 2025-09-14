#!/usr/bin/env python3
"""
Example usage of OpenAILikeResponses for OpenAI-compatible APIs with /responses support.

This example demonstrates how to use the OpenAILikeResponses class to interact with
OpenAI-compatible servers that support the /responses API endpoint.
"""

from llama_index.llms.openai_like import OpenAILikeResponses
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool


def search_web(query: str) -> str:
    """Search the web for information about a query."""
    # This is a mock implementation - in reality, you'd call a real search API
    return f"Search results for '{query}': Found relevant information about {query}."


def main():
    """Demonstrate usage of OpenAILikeResponses."""
    print("OpenAILikeResponses Example")
    print("=" * 40)
    
    # Create an OpenAILikeResponses instance
    # Replace with your actual API endpoint and key
    llm = OpenAILikeResponses(
        model="gpt-4o-mini",  # Use whatever model your API supports
        api_base="https://your-openai-compatible-api.com/v1",  # Your API endpoint
        api_key="your-api-key-here",  # Your API key
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
        
        # Responses-specific parameters
        max_output_tokens=1000,
        instructions="You are a helpful assistant that provides accurate and concise answers.",
        track_previous_responses=True,
        built_in_tools=[{"type": "web_search"}],  # Enable built-in web search
        user="example_user",
    )
    
    print(f"✓ Created {llm.class_name()}")
    print(f"✓ Model: {llm.model}")
    print(f"✓ API Base: {llm.api_base}")
    print(f"✓ Context Window: {llm.context_window}")
    print(f"✓ Max Output Tokens: {llm.max_output_tokens}")
    print(f"✓ Track Previous Responses: {llm.track_previous_responses}")
    print()
    
    # Example 1: Simple chat completion
    print("Example 1: Simple Chat Completion")
    print("-" * 30)
    
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello! Can you tell me about Python programming?")
    ]
    
    print("Note: This would make an actual API call to your OpenAI-compatible server")
    print("Messages to send:", [{"role": msg.role, "content": msg.content} for msg in messages])
    print()
    
    # Example 2: Function calling with tools
    print("Example 2: Function Calling with Tools")
    print("-" * 30)
    
    # Create a tool
    search_tool = FunctionTool.from_defaults(
        fn=search_web,
        name="search_web",
        description="Search the web for information"
    )
    
    # Prepare chat with tools
    tool_chat_params = llm._prepare_chat_with_tools(
        tools=[search_tool],
        user_msg="Search for the latest developments in artificial intelligence",
        tool_required=False
    )
    
    print(f"✓ Prepared chat with {len(tool_chat_params['tools'])} tools")
    print("Tool specifications:")
    for i, tool_spec in enumerate(tool_chat_params['tools']):
        print(f"  {i+1}. {tool_spec['name']}: {tool_spec.get('description', 'No description')}")
    
    print(f"✓ Messages prepared: {len(tool_chat_params['messages'])}")
    print(f"✓ Tool choice: {tool_chat_params.get('tool_choice', 'auto')}")
    print()
    
    # Example 3: Model kwargs for responses API
    print("Example 3: Model Kwargs for Responses API")
    print("-" * 30)
    
    model_kwargs = llm._get_model_kwargs(
        tools=[{"type": "function", "name": "custom_tool"}]
    )
    
    print("Generated model kwargs:")
    for key, value in model_kwargs.items():
        if key == 'tools':
            print(f"  {key}: {len(value)} tools")
        elif isinstance(value, str) and len(value) > 50:
            print(f"  {key}: {value[:47]}...")
        else:
            print(f"  {key}: {value}")
    print()
    
    print("Example completed successfully!")
    print("To use this with a real API, update the api_base and api_key parameters.")


if __name__ == "__main__":
    main()