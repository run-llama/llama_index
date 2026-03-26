"""
Example: Superlinked + LlamaIndex custom retriever (Steam games).

This example shows how to:
- Build a Superlinked pipeline (schema, space, index, app)
- Define a parameterized Superlinked QueryDescriptor using sl.Param("query_text")
- Inject the Superlinked App and QueryDescriptor into the LlamaIndex retriever
- Retrieve nodes with real similarity scores and optional engine usage

Run:
  python examples/steam_games_example.py [--csv /path/to/games.csv] [--top_k 5] [--query "strategic sci-fi game"]

If --csv is omitted, a tiny in-memory sample dataset is used.
"""
import argparse
from typing import List, Optional

import pandas as pd

import superlinked.framework as sl
from llama_index.retrievers.superlinked import SuperlinkedRetriever

try:
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer
except Exception:
    RetrieverQueryEngine = None  # type: ignore
    get_response_synthesizer = None  # type: ignore


def build_dataframe(csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # Minimal fallback sample
        df = pd.DataFrame(
            [
                {
                    "game_number": 1,
                    "name": "Star Tactics",
                    "desc_snippet": "Turn-based strategy in deep space.",
                    "game_details": "Tactical combat, fleet management",
                    "languages": "en",
                    "genre": "Strategy, Sci-Fi",
                    "game_description": "Engage in strategic battles among the stars.",
                    "original_price": 29.99,
                    "discount_price": 19.99,
                },
                {
                    "game_number": 2,
                    "name": "Wizard Party",
                    "desc_snippet": "Co-op party game with spells.",
                    "game_details": "Local co-op, party",
                    "languages": "en",
                    "genre": "Party, Casual, Magic",
                    "game_description": "Cast spells with friends in chaotic party modes.",
                    "original_price": 14.99,
                    "discount_price": 9.99,
                },
            ]
        )

    required = [
        "game_number",
        "name",
        "desc_snippet",
        "game_details",
        "languages",
        "genre",
        "game_description",
        "original_price",
        "discount_price",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["combined_text"] = (
        df["name"].astype(str)
        + " "
        + df["desc_snippet"].astype(str)
        + " "
        + df["genre"].astype(str)
        + " "
        + df["game_details"].astype(str)
        + " "
        + df["game_description"].astype(str)
    )
    return df


def build_superlinked_app(df: pd.DataFrame):
    class GameSchema(sl.Schema):
        id: sl.IdField
        name: sl.String
        desc_snippet: sl.String
        game_details: sl.String
        languages: sl.String
        genre: sl.String
        game_description: sl.String
        original_price: sl.Float
        discount_price: sl.Float
        combined_text: sl.String

    game = GameSchema()

    text_space = sl.TextSimilaritySpace(
        text=game.combined_text,
        model="sentence-transformers/all-mpnet-base-v2",
    )
    index = sl.Index([text_space])

    parser = sl.DataFrameParser(
        game,
        mapping={
            game.id: "game_number",
            game.name: "name",
            game.desc_snippet: "desc_snippet",
            game.game_details: "game_details",
            game.languages: "languages",
            game.genre: "genre",
            game.game_description: "game_description",
            game.original_price: "original_price",
            game.discount_price: "discount_price",
            game.combined_text: "combined_text",
        },
    )

    source = sl.InMemorySource(schema=game, parser=parser)
    executor = sl.InMemoryExecutor(sources=[source], indices=[index])
    app = executor.run()

    source.put([df])

    # Build parameterized query using sl.Param("query_text")
    query = (
        sl.Query(index)
        .find(game)
        .similar(text_space, sl.Param("query_text"))
        .select(
            [
                game.id,
                game.name,
                game.desc_snippet,
                game.game_details,
                game.languages,
                game.genre,
                game.game_description,
                game.original_price,
                game.discount_price,
            ]
        )
        # Do not set .limit() here; the retriever will cap results via k
    )

    return app, query, game


def run_demo(csv_path: Optional[str], top_k: int, query_text: str) -> None:
    df = build_dataframe(csv_path)
    app, query_descriptor, game = build_superlinked_app(df)

    # Inject Superlinked App and QueryDescriptor into the LlamaIndex retriever
    retriever = SuperlinkedRetriever(
        sl_client=app,
        sl_query=query_descriptor,
        page_content_field="desc_snippet",
        query_text_param="query_text",
        metadata_fields=[
            "id",
            "name",
            "genre",
            "game_details",
            "languages",
            "game_description",
            "original_price",
            "discount_price",
        ],
        top_k=top_k,
    )

    print(f"\nRetrieving for: {query_text!r}")
    nodes = retriever.retrieve(query_text)
    for i, nws in enumerate(nodes, 1):
        print(f"#{i} score={nws.score:.4f} text={nws.node.text!r}")
        print(f"   metadata: {nws.node.metadata}")

    # Optional: use LlamaIndex query engine if packages are available
    if RetrieverQueryEngine and get_response_synthesizer:
        print("\nBuilding RetrieverQueryEngine...")
        try:
            engine = RetrieverQueryEngine(
                retriever=retriever, response_synthesizer=get_response_synthesizer()
            )
            response = engine.query(query_text)
            print("\nEngine response:", response)
        except Exception as e:
            print("Engine invocation failed (likely missing LLM setup):", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to games CSV file")
    parser.add_argument("--top_k", type=int, default=5, help="Max results to return")
    parser.add_argument(
        "--query",
        type=str,
        default="strategic sci-fi game",
        help="Query text",
    )
    args = parser.parse_args()
    run_demo(args.csv, args.top_k, args.query)
