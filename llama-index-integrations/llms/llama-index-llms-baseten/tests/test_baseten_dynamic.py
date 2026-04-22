#!/usr/bin/env python3
"""
Test script for Baseten LLM dynamic validation implementation.
This demonstrates the new dynamic model validation pattern adapted from NVIDIA.
"""

import os
import sys

# Add the Baseten LLM integration to the path
sys.path.insert(
    0,
    "/Users/alexker/code/llama_index/llama-index-integrations/llms/llama-index-llms-baseten",
)


def test_baseten_dynamic_validation():
    """Test the dynamic validation features."""
    print("🧪 Testing Baseten Dynamic Validation Implementation")
    print("=" * 60)

    try:
        from llama_index.llms.baseten import Baseten
        from llama_index.llms.baseten.utils import Model, get_supported_models

        print("✅ Successfully imported Baseten with dynamic validation")
        print()

        # Test 1: Static model list (existing functionality)
        print("📋 Test 1: Static Model List")
        static_models = get_supported_models()
        print(f"   Found {len(static_models)} static models:")
        for i, model in enumerate(static_models[:5]):  # Show first 5
            print(f"   {i + 1}. {model}")
        if len(static_models) > 5:
            print(f"   ... and {len(static_models) - 5} more")
        print()

        # Test 2: Model class functionality
        print("🔧 Test 2: Model Class")
        test_model = Model(id="test-model")
        print(
            f"   Created model: {test_model.id} (type: {test_model.model_type}, client: {test_model.client})"
        )
        print()

        # Test 3: Dynamic validation with API key
        print("🔑 Test 3: Dynamic Validation")

        if os.getenv("BASETEN_API_KEY"):
            print("   API key found - testing live dynamic validation")
            try:
                # Use a known valid model from the static list
                valid_model = static_models[0]
                print(f"   Creating Baseten LLM with model: {valid_model}")

                llm = Baseten(model_id=valid_model, model_apis=True)
                print("   ✅ Successfully created Baseten LLM with dynamic validation")

                # Test available_models property
                print("   📡 Testing available_models property...")
                try:
                    available = llm.available_models
                    print(f"   ✅ Fetched {len(available)} models dynamically")
                    print(f"   First few available models:")
                    for i, model in enumerate(available[:3]):
                        print(f"      {i + 1}. {model.id}")

                    # Compare static vs dynamic
                    dynamic_ids = {model.id for model in available}
                    static_ids = set(static_models)

                    if dynamic_ids != static_ids:
                        print("   📊 Differences between static and dynamic lists:")
                        only_dynamic = dynamic_ids - static_ids
                        only_static = static_ids - dynamic_ids

                        if only_dynamic:
                            print(
                                f"      New models (dynamic only): {list(only_dynamic)[:3]}"
                            )
                        if only_static:
                            print(
                                f"      Removed models (static only): {list(only_static)[:3]}"
                            )
                    else:
                        print("   📊 Static and dynamic lists match perfectly")

                except Exception as e:
                    print(f"   ⚠️  Dynamic model fetching failed (using fallback): {e}")

            except Exception as e:
                print(f"   ❌ Failed to create Baseten LLM: {e}")
        else:
            print("   ⚠️  No BASETEN_API_KEY found - skipping live API tests")
            print(
                "   Set BASETEN_API_KEY environment variable to test dynamic validation"
            )
        print()

        # Test 4: Error handling with invalid model
        print("🚫 Test 4: Error Handling")
        print("   Testing with invalid model name...")
        try:
            llm = Baseten(model_id="invalid-model-name-12345", model_apis=True)
            print("   ❌ Should have failed with invalid model")
        except ValueError as e:
            error_msg = str(e)
            print(f"   ✅ Correctly caught validation error")
            print(f"   Error message: {error_msg[:80]}...")

            # Check if error message includes suggestions
            if "Did you mean" in error_msg or "Available models" in error_msg:
                print("   ✅ Error message includes helpful suggestions")
            else:
                print("   ⚠️  Error message could be more helpful")
        except Exception as e:
            print(f"   ⚠️  Unexpected error type: {type(e).__name__}: {e}")
        print()

        # Test 5: Dedicated deployment mode (no dynamic validation)
        print("🏗️  Test 5: Dedicated Deployment Mode")
        print("   Testing with model_apis=False (dedicated deployment)...")
        try:
            dedicated_llm = Baseten(model_id="12345678", model_apis=False)
            print("   ✅ Successfully created dedicated deployment LLM (no validation)")

            available_dedicated = dedicated_llm.available_models
            print(
                f"   Available models for dedicated: {len(available_dedicated)} models"
            )
            if available_dedicated:
                print(f"   Models: {[m.id for m in available_dedicated]}")

        except Exception as e:
            print(f"   ❌ Failed to create dedicated LLM: {e}")
        print()

        print("🎉 All tests completed!")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the llama_index directory")
        print("And that you have the necessary dependencies installed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


def print_usage():
    """Print usage instructions."""
    print("📖 How to run this test:")
    print()
    print("1. Set up your environment:")
    print("   export BASETEN_API_KEY='your-api-key-here'")
    print()
    print("2. Run the test:")
    print("   cd /Users/alexker/code/llama_index")
    print("   python test_baseten_dynamic.py")
    print()
    print("3. Optional: Run without API key (limited testing):")
    print("   python test_baseten_dynamic.py --no-api")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print_usage()
    else:
        test_baseten_dynamic_validation()
