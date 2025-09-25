#!/usr/bin/env python3
"""
Example demonstrating structured output with OpenAILikeResponses.

This example shows how to use the OpenAILikeResponses class with structured output
to extract structured data from LLM responses using Pydantic models.
"""

from pydantic import BaseModel, Field
from llama_index.llms.openai_like import OpenAILikeResponses
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class PersonInfo(BaseModel):
    """Pydantic model for structured person information."""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years") 
    city: str = Field(description="The city where the person lives")
    profession: str = Field(description="The person's profession or job")


class CityInfo(BaseModel):
    """Pydantic model for structured city information."""
    capital: str = Field(description="The capital city")
    country: str = Field(description="The country name")
    population: str = Field(description="The population of the city/country")
    area: str = Field(description="The area of the city/country")
    currency: str = Field(description="The official currency")
    language: str = Field(description="The primary language spoken")
    time_zone: str = Field(description="The time zone")
    government_type: str = Field(description="The type of government")
    independence_year: str = Field(description="The year of independence")
    religion: str = Field(description="The predominant religion")


def main():
    """Demonstrate structured output functionality."""
    print("=== OpenAILikeResponses Structured Output Example ===\n")
    
    # Initialize the LLM
    llm = OpenAILikeResponses(
        model="/models/gpt-oss-120b",
        api_base="http://your-host:8021/v1",
        api_key="your-api-key",
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0.7,
    )
    
    print("1. Creating structured LLM for PersonInfo...")
    person_llm = llm.as_structured_llm(PersonInfo)
    
    print("2. Example: Extract person information")
    print("   Input: 'Tell me about Alice, a 28-year-old software engineer in San Francisco'")
    
    try:
        response = person_llm.complete(
            "Tell me about Alice, a 28-year-old software engineer in San Francisco"
        )
        
        # The response.raw contains the structured Pydantic object
        person_data = response.raw
        print(f"   Output:")
        print(f"   - Name: {person_data.name}")
        print(f"   - Age: {person_data.age}")
        print(f"   - City: {person_data.city}")
        print(f"   - Profession: {person_data.profession}")
        
    except Exception as e:
        print(f"   Error with PersonInfo example: {e}")
    
    print("\n" + "="*50 + "\n")
    
    print("3. Creating structured LLM for CityInfo...")
    city_llm = llm.as_structured_llm(CityInfo)
    
    print("4. Example: Extract detailed city/country information")
    print("   Input: 'Write a short story about Paris'")
    
    try:
        response = city_llm.complete("Write a short story about Paris")
        
        # The response.raw contains the structured Pydantic object  
        city_data = response.raw
        print(f"   Output:")
        print(f"   - Capital: {city_data.capital}")
        print(f"   - Country: {city_data.country}")
        print(f"   - Population: {city_data.population}")
        print(f"   - Area: {city_data.area}")
        print(f"   - Currency: {city_data.currency}")
        print(f"   - Language: {city_data.language}")
        print(f"   - Time Zone: {city_data.time_zone}")
        print(f"   - Government: {city_data.government_type}")
        print(f"   - Independence: {city_data.independence_year}")
        print(f"   - Religion: {city_data.religion}")
        
    except Exception as e:
        print(f"   Error with CityInfo example: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("\nNOTE: Make sure to:")
    print("- Update the api_base URL to point to your OpenAI-compatible server")
    print("- Set the correct api_key for your server")
    print("- Ensure your server supports the /responses API endpoint")
    print("- For servers that don't support structured output, the implementation")
    print("  will fall back to function calling for structured data extraction")


if __name__ == "__main__":
    main()