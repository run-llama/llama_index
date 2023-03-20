"""Utilities for Spider module."""

import json
import os
from typing import Dict, Tuple, Union

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine

from gpt_index import GPTSQLStructStoreIndex, LLMPredictor, SQLDatabase


def load_examples(spider_dir: str) -> Tuple[list, list]:
    """Load examples."""
    with open(os.path.join(spider_dir, "train_spider.json"), "r") as f:
        train_spider = json.load(f)
    with open(os.path.join(spider_dir, "train_others.json"), "r") as f:
        train_others = json.load(f)
    with open(os.path.join(spider_dir, "dev.json"), "r") as f:
        dev = json.load(f)
    return train_spider + train_others, dev


def create_indexes(
    spider_dir: str, llm: Union[ChatOpenAI, OpenAI]
) -> Dict[str, GPTSQLStructStoreIndex]:
    """Create indexes for all databases."""
    # Create all necessary SQL database objects.
    databases = {}
    for db_name in os.listdir(os.path.join(spider_dir, "database")):
        db_path = os.path.join(spider_dir, "database", db_name, db_name + ".sqlite")
        if not os.path.exists(db_path):
            continue
        engine = create_engine("sqlite:///" + db_path)
        databases[db_name] = SQLDatabase(engine=engine)
        # Test connection.
        engine.execute("select name from sqlite_master where type = 'table'").fetchone()

    llm_predictor = LLMPredictor(llm=llm)
    llm_indexes = {}
    for db_name, db in databases.items():
        llm_indexes[db_name] = GPTSQLStructStoreIndex(
            llm_predictor=llm_predictor,
            sql_database=db,
        )
    return llm_indexes
