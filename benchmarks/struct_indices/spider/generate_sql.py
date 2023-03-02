import argparse
import json
import logging
import os

from langchain import OpenAI
from sqlalchemy import create_engine
from tqdm import tqdm

from gpt_index import GPTSQLStructStoreIndex, LLMPredictor, SQLDatabase

logging.getLogger("root").setLevel(logging.WARNING)


def _generate_sql(
    llama_index: GPTSQLStructStoreIndex,
    nl_query_text: str,
) -> str:
    response = llama_index.query(nl_query_text, mode="default")
    return response.extra_info["sql_query"].replace("\n", " ").strip()


def generate_sql(llama_indexes: dict, examples: list, output_file: str):
    with open(output_file, "w") as f:
        for example in tqdm(examples, desc=f"Generating {output_file}"):
            db_name = example["db_id"]
            nl_query_text = example["question"]
            sql_query = _generate_sql(llama_indexes[db_name], nl_query_text)
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
        "to be compared wit the *_gold.sql files in the input directory.",
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
    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-davinci-003")
    )
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
