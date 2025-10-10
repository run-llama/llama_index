#!/usr/bin/env python3
"""
Demonstration of the PolarsQueryEngine

This script shows how to use the new PolarsQueryEngine alongside 
the existing PandasQueryEngine for DataFrame-based querying.
"""

import polars as pl
import pandas as pd
from llama_index.experimental.query_engine import PolarsQueryEngine, PandasQueryEngine
from llama_index.core.llms.mock import MockLLM

def demo_polars_query_engine():
    """Demonstrate PolarsQueryEngine functionality."""
    print("=== PolarsQueryEngine Demo ===")
    
    # Create sample data
    polars_df = pl.DataFrame({
        "city": ["Toronto", "Tokyo", "Berlin", "New York", "London"],
        "population": [2930000, 13960000, 3645000, 8336817, 8982000],
        "country": ["Canada", "Japan", "Germany", "USA", "UK"],
        "continent": ["North America", "Asia", "Europe", "North America", "Europe"]
    })
    
    print("Sample Polars DataFrame:")
    print(polars_df)
    print()
    
    # Mock LLM that returns specific Polars queries
    class PolarsQueryMockLLM(MockLLM):
        def predict(self, *args, **kwargs):
            query_str = kwargs.get("query_str", "").lower()
            if "population" in query_str and "greater" in query_str:
                return "df.filter(pl.col('population') > 5000000)"
            elif "cities" in query_str and "europe" in query_str:
                return "df.filter(pl.col('continent') == 'Europe').select(['city', 'country'])"
            elif "average" in query_str and "population" in query_str:
                return "df.select(pl.col('population').mean())"
            elif "group" in query_str:
                return "df.group_by('continent').agg(pl.col('population').sum())"
            else:
                return "df.head(3)"
    
    # Create query engine
    llm = PolarsQueryMockLLM()
    query_engine = PolarsQueryEngine(df=polars_df, llm=llm, verbose=True)
    
    # Test queries
    test_queries = [
        "Show cities with population greater than 5 million",
        "List cities in Europe with their countries", 
        "What is the average population?",
        "Group cities by continent and sum population",
        "Show first few rows"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        try:
            response = query_engine.query(query)
            print(f"📊 Result:\n{response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


def demo_comparison():
    """Show comparison between Pandas and Polars query engines."""
    print("\n=== Pandas vs Polars Comparison ===")
    
    # Create equivalent dataframes
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28], 
        "department": ["Engineering", "Sales", "Engineering", "Marketing"],
        "salary": [75000, 65000, 85000, 70000]
    }
    
    pandas_df = pd.DataFrame(data)
    polars_df = pl.DataFrame(data)
    
    print("Data (same for both):")
    print(pandas_df)
    print()
    
    # Mock LLMs for both engines
    class PandasSalaryQueryMockLLM(MockLLM):
        def predict(self, *args, **kwargs):
            return "df[df['salary'] > 70000]"
    
    class PolarsSalaryQueryMockLLM(MockLLM):
        def predict(self, *args, **kwargs):
            return "df.filter(pl.col('salary') > 70000)"
    
    # Create both query engines
    pandas_llm = PandasSalaryQueryMockLLM()
    polars_llm = PolarsSalaryQueryMockLLM()
    
    pandas_engine = PandasQueryEngine(df=pandas_df, llm=pandas_llm, verbose=False)
    polars_engine = PolarsQueryEngine(df=polars_df, llm=polars_llm, verbose=False)
    
    query = "Show employees with salary greater than 70000"
    
    print(f"🔍 Query: {query}")
    print("-" * 50)
    
    print("📊 Pandas Result:")
    pandas_response = pandas_engine.query(query)
    print(pandas_response)
    print()
    
    print("📊 Polars Result:")
    polars_response = polars_engine.query(query)
    print(polars_response)
    print()


if __name__ == "__main__":
    demo_polars_query_engine()
    demo_comparison()
    
    print("✅ PolarsQueryEngine demonstration completed successfully!")
    print("\n🎉 Key Features of PolarsQueryEngine:")
    print("- Expression-based querying with Polars syntax")
    print("- Secure execution with sandboxing (same as PandasQueryEngine)")
    print("- Support for complex operations: filtering, grouping, aggregations")
    print("- Compatible with LlamaIndex ecosystem")
    print("- Fast columnar operations with Arrow backend")