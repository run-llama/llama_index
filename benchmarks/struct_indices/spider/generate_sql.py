"""Generate SQL queries using LlamaIndex."""
import argparse
import json
import logging
import os
import re

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import BaseLanguageModel
from sqlalchemy import create_engine
from tqdm import tqdm

from gpt_index import GPTSQLStructStoreIndex, LLMPredictor, SQLDatabase

logging.getLogger("root").setLevel(logging.WARNING)


_spaces = re.compile(r"\s+")
_newlines = re.compile(r"\n+")


def _generate_sql(
    llama_index: GPTSQLStructStoreIndex,
    nl_query_text: str,
) -> str:
    """Generate SQL query for the given NL query text."""
    response = llama_index.query(nl_query_text)
    if (
        response.extra_info is None
        or "sql_query" not in response.extra_info
        or response.extra_info["sql_query"] is None
    ):
        raise RuntimeError("No SQL query generated.")
    query = response.extra_info["sql_query"]
    # Remove newlines and extra spaces.
    query = _newlines.sub(" ", query)
    query = _spaces.sub(" ", query)
    return query.strip()


def generate_sql(llama_indexes: dict, examples: list, output_file: str) -> None:
    """Generate SQL queries for the given examples and write them to the output file."""
    with open(output_file, "w") as f:
        for example in tqdm(examples, desc=f"Generating {output_file}"):
            db_name = example["db_id"]
            nl_query_text = example["question"]
            try:
                sql_query = _generate_sql(llama_indexes[db_name], nl_query_text)
            except Exception as e:
                print(
                    f"Failed to generate SQL query for question: "
                    f"{example['question']} on database: {example['db_id']}."
                )
                print(e)
                sql_query = "ERROR"
            f.write(sql_query + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SQL queries using LlamaIndex."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the spider dataset directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory of generated SQL files,"
        " one query on each line, "
        "to be compared with the *_gold.sql files in the input directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "code-davinci-002"],
        required=True,
        help="The model to use for generating SQL queries.",
    )
    args = parser.parse_args()

    # Create the output directory if it does not exist.
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the Spider dataset from the input directory.
    with open(os.path.join(args.input, "train_spider.json"), "r") as f:
        train_spider = json.load(f)
    with open(os.path.join(args.input, "train_others.json"), "r") as f:
        train_others = json.load(f)
    with open(os.path.join(args.input, "dev.json"), "r") as f:
        dev = json.load(f)

    # Create all necessary SQL database objects.
    databases = {}
    for db in train_spider + train_others + dev:
        db_name = db["db_id"]
        if db_name in databases:
            continue
        db_path = os.path.join(args.input, "database", db_name, db_name + ".sqlite")
        engine = create_engine("sqlite:///" + db_path)
        databases[db_name] = (SQLDatabase(engine=engine), engine)

    # Create the LlamaIndexes for all databases.
    if args.model in ["gpt-3.5-turbo", "gpt-4"]:
        llm: BaseLanguageModel = ChatOpenAI(model=args.model, temperature=0)
    else:
        llm = OpenAI(model=args.model, temperature=0)
    llm_predictor = LLMPredictor(llm=llm)
    llm_indexes = {}
    for db_name, (db, engine) in databases.items():
        # Get the name of the first table in the database.
        # This is a hack to get a table name for the index, which can use any
        # table in the database.
        table_name = engine.execute(
            "select name from sqlite_master where type = 'table'"
        ).fetchone()[0]
        llm_indexes[db_name] = GPTSQLStructStoreIndex(
            documents=[],
            llm_predictor=llm_predictor,
            sql_database=db,
            table_name=table_name,
        )

    # Generate SQL queries.
    generate_sql(
        llama_indexes=llm_indexes,
        examples=train_spider + train_others,
        output_file=os.path.join(args.output, "train_pred.sql"),
    )
    generate_sql(
        llama_indexes=llm_indexes,
        examples=dev,
        output_file=os.path.join(args.output, "dev_pred.sql"),
    )
