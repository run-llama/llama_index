#!/usr/bin/env python3
"""Local test script for SERPEX tool integration."""

import os

from llama_index.tools.serpex import SerpexToolSpec

# Test 1: Check initialization
print("Test 1: Initializing SERPEX tool...")
try:
    # You need to set your SERPEX_API_KEY environment variable
    api_key = os.environ.get("SERPEX_API_KEY")
    if not api_key:
        print("⚠️  Warning: SERPEX_API_KEY not set. Please set it:")
        print('   export SERPEX_API_KEY="your_api_key"')
        print("\nTrying with dummy key for structure test...")
        tool = SerpexToolSpec(api_key="dummy_key_for_testing")
        print("✅ Tool initialization works!")
    else:
        tool = SerpexToolSpec(api_key=api_key)
        print("✅ Tool initialized with real API key!")

        # Test 2: Basic search
        print("\nTest 2: Testing basic search...")
        results = tool.search("LlamaIndex tutorial", num_results=3)
        print(f"✅ Search returned {len(results)} results (as Document objects):")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(result.text[:500])  # Print first 500 chars

        # Test 3: Check tool list conversion
        print("\n" + "=" * 60)
        print("Test 4: Testing tool list conversion...")
        tool_list = tool.to_tool_list()
        print(f"✅ Tool list has {len(tool_list)} tools:")
        for t in tool_list:
            print(f"   - {t.metadata.name}: {t.metadata.description[:60]}...")

        print("\n" + "=" * 60)
        print("🎉 All tests passed! SERPEX integration is working!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
