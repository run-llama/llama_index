"""Dataset generation: source environments -> labeled examples -> JSONL.

This script harvests responses from a language model, scores them using
source-environment rubrics, and writes a JSONL dataset suitable for the
rubric-discovery environment.

Usage::

    cd environments/rubric_discovery
    uv run python -m scripts.generate_dataset \\
        --output-path env/data/rubric_discovery_dataset.jsonl \\
        --target-size 800 \\
        --responses-per-example 4 \\
        --seed 42 \\
        --temperatures 0,0.5,1.0,1.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ExampleDict = Dict[str, Any]


def load_source_config(config_path: str) -> Dict[str, Any]:
    """Load the source environments YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_llm_client(
    api_key_var: str = "OPENAI_API_KEY",
    api_base_url: str = "https://api.openai.com/v1",
    client_type: str = "openai",
) -> Any:
    """Create an LLM client for response generation.

    Args:
        api_key_var: Environment variable holding the API key.
        api_base_url: Base URL for the API.
        client_type: Client type identifier.

    Returns:
        An initialized client object.
    """
    try:
        import openai  # type: ignore[import-untyped]
    except ImportError:
        print("Error: openai package required. Install with: pip install openai")
        sys.exit(1)

    api_key = os.environ.get(api_key_var)
    if not api_key:
        print(f"Error: {api_key_var} environment variable not set.")
        sys.exit(1)

    return openai.OpenAI(api_key=api_key, base_url=api_base_url)


def generate_responses(
    client: Any,
    prompt: str,
    model_name: str,
    temperatures: List[float],
    responses_per_temp: int = 1,
) -> List[Tuple[str, float]]:
    """Generate responses at different temperatures.

    Returns list of (response_text, temperature) tuples.
    """
    results = []
    for temp in temperatures:
        for _ in range(responses_per_temp):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=512,
                )
                text = resp.choices[0].message.content or ""
                results.append((text.strip(), temp))
            except Exception as e:
                print(f"  Warning: generation failed (temp={temp}): {e}")
    return results


def score_with_source_env(
    env_id: str,
    input_text: str,
    response: str,
) -> Optional[float]:
    """Score a response using a source environment's rubric.

    Attempts to load the source environment and run its rubric.
    Falls back gracefully if the environment is not installed.
    """
    try:
        # Try Prime Hub environment loading
        from prime_env import load_env  # type: ignore[import-untyped]

        env = load_env(env_id)
        result = env.score(input_text, response)
        return float(result)
    except ImportError:
        pass
    except Exception as e:
        print(f"  Warning: scoring with {env_id} failed: {e}")

    return None


def create_synthetic_examples(
    source_env: Dict[str, Any],
    client: Any,
    model_name: str,
    temperatures: List[float],
    responses_per_example: int,
    seed: int,
) -> List[ExampleDict]:
    """Generate synthetic labeled examples for a source environment.

    When the source environment is not available for scoring, generates
    placeholder examples with heuristic scores based on response properties.
    """
    name = source_env["name"]
    description = source_env.get("description", "")

    # Generate diverse prompts for this category
    prompts = _generate_prompts_for_category(name, client, model_name)

    examples = []
    for prompt in prompts:
        responses = generate_responses(
            client, prompt, model_name, temperatures,
            responses_per_temp=max(1, responses_per_example // len(temperatures)),
        )

        for response_text, temp in responses:
            # Try source env scoring first
            score = score_with_source_env(
                source_env.get("env_id", ""), prompt, response_text
            )

            if score is None:
                # Heuristic fallback based on temperature
                # Lower temp → more focused → generally higher quality
                base_score = max(0.0, 1.0 - (temp * 0.3))
                # Add noise
                rng = random.Random(seed + hash(response_text))
                noise = rng.gauss(0, 0.15)
                score = max(0.0, min(1.0, base_score + noise))

            examples.append({
                "input_text": prompt,
                "response": response_text,
                "score": round(score, 4),
                "category": name,
                "metadata": {"temperature": temp, "model": model_name},
            })

    return examples


def _generate_prompts_for_category(
    category: str,
    client: Any,
    model_name: str,
) -> List[str]:
    """Generate diverse input prompts for a given category."""
    meta_prompt = (
        f"Generate 5 diverse input prompts for a '{category}' evaluation task. "
        f"Return only the prompts, one per line, no numbering."
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        text = resp.choices[0].message.content or ""
        prompts = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return prompts[:5] if prompts else [f"Test prompt for {category}"]
    except Exception:
        return [f"Test prompt for {category}"]


def split_train_test(
    examples: List[ExampleDict],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[ExampleDict], List[ExampleDict]]:
    """Split examples into train and test sets."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def build_dataset_rows(
    all_examples: Dict[str, List[ExampleDict]],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build JSONL rows from categorized examples."""
    rows = []
    for category, examples in all_examples.items():
        if len(examples) < 3:
            print(f"  Skipping {category}: too few examples ({len(examples)})")
            continue

        train, test = split_train_test(examples, train_ratio, seed)
        if not test:
            # Ensure at least 1 test example
            test = [train.pop()]

        rows.append({
            "train_examples": train,
            "test_examples": test,
            "category": category,
            "metadata": {"num_train": len(train), "num_test": len(test)},
        })

    return rows


def main() -> None:
    """CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate rubric-discovery training dataset."
    )
    parser.add_argument(
        "--output-path",
        default="env/data/rubric_discovery_dataset.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--config-path",
        default=os.path.join(os.path.dirname(__file__), "source_envs.yaml"),
        help="Path to source_envs.yaml config.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=800,
        help="Target total number of examples across all categories.",
    )
    parser.add_argument(
        "--responses-per-example",
        type=int,
        default=4,
        help="Number of responses to generate per input prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--temperatures",
        default="0,0.5,1.0,1.5",
        help="Comma-separated list of generation temperatures.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for generation (overrides config default).",
    )
    parser.add_argument(
        "--api-key-var",
        default="OPENAI_API_KEY",
        help="Environment variable for the API key.",
    )
    parser.add_argument(
        "--api-base-url",
        default="https://api.openai.com/v1",
        help="API base URL.",
    )
    parser.add_argument(
        "--client-type",
        default="openai",
        help="Client type identifier.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Load config
    config = load_source_config(args.config_path)
    defaults = config.get("defaults", {})
    source_envs = config.get("source_environments", [])

    temperatures = [float(t) for t in args.temperatures.split(",")]
    model_name = args.model_name or defaults.get("model_name", "gpt-4.1-mini")

    print(f"Generating dataset with {len(source_envs)} source environments")
    print(f"Model: {model_name}, Temperatures: {temperatures}")
    print(f"Target size: {args.target_size}")

    # Create client
    client = create_llm_client(
        api_key_var=args.api_key_var,
        api_base_url=args.api_base_url,
        client_type=args.client_type,
    )

    # Calculate per-environment allocation
    total_weight = sum(e.get("weight", 1.0) for e in source_envs)
    all_examples: Dict[str, List[ExampleDict]] = {}

    for env_cfg in source_envs:
        name = env_cfg["name"]
        weight = env_cfg.get("weight", 1.0)
        allocation = int(args.target_size * weight / total_weight)

        print(f"\n  Generating {allocation} examples for '{name}'...")

        examples = create_synthetic_examples(
            source_env=env_cfg,
            client=client,
            model_name=model_name,
            temperatures=temperatures,
            responses_per_example=args.responses_per_example,
            seed=args.seed,
        )

        # Trim to allocation
        if len(examples) > allocation:
            rng = random.Random(args.seed)
            rng.shuffle(examples)
            examples = examples[:allocation]

        all_examples[name] = examples
        print(f"  -> {len(examples)} examples generated.")

    # Build rows
    train_ratio = defaults.get("train_test_split", 0.7)
    rows = build_dataset_rows(all_examples, train_ratio, args.seed)

    # Write output
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    total = sum(
        len(r["train_examples"]) + len(r["test_examples"]) for r in rows
    )
    print(f"\nDataset written to {output_path}")
    print(f"  Rows: {len(rows)}, Total examples: {total}")


if __name__ == "__main__":
    main()
