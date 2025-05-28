"""Spider evaluation script."""

import argparse
import ast
import json
import logging
import os
from typing import Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import Response
from llama_index.indices.struct_store.sql import SQLQueryMode, SQLStructStoreIndex
from llama_index.llms.openai import OpenAI
from spider_utils import create_indexes, load_examples
from tqdm import tqdm

logging.getLogger("root").setLevel(logging.WARNING)

answer_template = (
    "Given an input question, first create a syntactically correct SQL "
    "query to run, then look at the results of the query and return the answer. "
    "Use the following format:\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n"
    "Question: {question}\n"
    "SQLQuery: {sql_query}\n"
    "SQLResult: {sql_result}"
    "Answer: "
)

match_template = """Given a question, a reference answer and a hypothesis answer, \
    determine if the hypothesis answer is correct. Use the following format:

Question: Question here
ReferenceAnswer: Reference answer here
HypothesisAnswer: Hypothesis answer here
HypothesisAnswerCorrect: true or false

Question: {question}
ReferenceAnswer: {reference_answer}
HypothesisAnswer: {hypothesis_answer}
HypothesisAnswerCorrect: """


def _answer(
    llm: OpenAI, question: str, sql_query: str, sql_result: Optional[str]
) -> str:
    prompt = answer_template.format(
        question=question, sql_query=sql_query, sql_result=sql_result
    )
    response = llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
    return response.message.content or ""


def _match(
    llm: OpenAI, question: str, reference_answer: str, hypothesis_answer: str
) -> bool:
    prompt = match_template.format(
        question=question,
        reference_answer=reference_answer,
        hypothesis_answer=hypothesis_answer,
    )
    response = llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
    content = response.message.content or ""
    return "true" in content.lower()


def _get_answers(
    llm: OpenAI,
    indexes: Dict[str, SQLStructStoreIndex],
    db_names: List[str],
    sql_queries: List[str],
    examples: List[dict],
    output_filename: str,
    use_cache: bool,
) -> List[dict]:
    if use_cache and os.path.exists(output_filename):
        with open(output_filename) as f:
            return json.load(f)

    results = []
    for db_name, sql_query, example in tqdm(
        list(zip(db_names, sql_queries, examples)),
        desc=f"Getting NL Answers to: {output_filename}",
    ):
        assert example["db_id"] == db_name
        question = example["question"]
        result = {
            "question": question,
            "sql_query": sql_query,
            "sql_result": None,
            "answer": None,
        }
        results.append(result)
        if sql_query.strip() == "ERROR":
            result["sql_result"] = "ERROR"
            result["answer"] = "ERROR"
        try:
            query_engine = indexes[db_name].as_query_engine(query_mode=SQLQueryMode.SQL)
            resp = query_engine.query(sql_query)
            assert isinstance(resp, Response)
            result["sql_result"] = resp.response
            if resp.response is None:
                result["answer"] = ""
            result["answer"] = _answer(llm, question, sql_query, resp.response)
        except Exception as e:
            print(f"Error encountered when answering question ({question}): {e}")
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)
    return results


def _match_answers(
    llm: OpenAI,
    gold_results: List[dict],
    pred_results: List[dict],
    examples: List[dict],
    output_filename: str,
) -> float:
    results = []
    for gold, pred, example in tqdm(
        list(zip(gold_results, pred_results, examples)),
        desc=f"Evaluating: {output_filename}",
    ):
        assert gold["question"] == example["question"]
        assert pred["question"] == example["question"]

        # Match execution results.
        if pred["sql_result"] is None or gold["sql_result"] is None:
            exec_match = None
        elif pred["sql_result"] == "ERROR":
            exec_match = False
        else:
            try:
                p_tuples = set(ast.literal_eval(pred["sql_result"]))
                g_tuples = set(ast.literal_eval(gold["sql_result"]))
                exec_match = p_tuples == g_tuples
            except Exception as e:
                print("Error encountered when parsing SQL result: ", e)
                exec_match = None

        # Match NL answers.
        if pred["answer"] is None or gold["answer"] is None:
            answer_match = None
        elif pred["answer"] == "ERROR":
            answer_match = False
        else:
            answer_match = _match(
                llm, example["question"], gold["answer"], pred["answer"]
            )

        results.append(
            {
                "db": example["db_id"],
                "exec_match": exec_match,
                "answer_match": answer_match,
                "gold": gold,
                "pred": pred,
            }
        )
    valid_results = [
        e
        for e in results
        if e["exec_match"] is not None and e["answer_match"] is not None
    ]
    answer_accuracy = sum(
        [e["exec_match"] or e["answer_match"] for e in valid_results]
    ) / float(len(valid_results))
    with open(output_filename, "w") as f:
        json.dump(
            {
                "answer_accuracy": answer_accuracy,
                "total": len(results),
                "valid": len(valid_results),
                "results": results,
            },
            f,
            indent=2,
        )
    return answer_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer accuracy of generated SQL queries by "
            "checking the NL answer generated from execution output."
        )
    )
    parser.add_argument(
        "--spider-dir", type=str, required=True, help="Path to the Spider directory."
    )
    parser.add_argument(
        "--predict-dir",
        type=str,
        required=True,
        help="Path to the directory of generated SQL files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-3.5-turbo"],
        help="The model used to perform evaluation.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Whether to use the cached results or not.",
    )
    args = parser.parse_args()

    # Create the LlamaIndexes for all databases.
    llm = OpenAI(model=args.model, temperature=0)

    # Load all examples.
    train, dev = load_examples(args.spider_dir)

    # Load all generated SQL queries.
    with open(os.path.join(args.predict_dir, "train_pred.sql")) as f:
        train_pred_sqls = f.readlines()
    with open(os.path.join(args.predict_dir, "dev_pred.sql")) as f:
        dev_pred_sqls = f.readlines()

    # Load all gold SQL queries and database names.
    train_dbs = []
    dev_dbs = []
    train_gold_sqls = []
    dev_gold_sqls = []
    with open(os.path.join(args.spider_dir, "train_gold.sql")) as f:
        for line in f.readlines():
            line_tokens = line.strip().split("\t")
            train_gold_sqls.append(line_tokens[0])
            train_dbs.append(line_tokens[1])
    with open(os.path.join(args.spider_dir, "dev_gold.sql")) as f:
        for line in f.readlines():
            line_tokens = line.strip().split("\t")
            dev_gold_sqls.append(line_tokens[0])
            dev_dbs.append(line_tokens[1])

    # Create Llama indexes on the databases.
    indexes = create_indexes(spider_dir=args.spider_dir, llm=llm)

    # Run SQL queries on the indexes and get NL answers.
    train_pred_results = _get_answers(
        llm,
        indexes,
        train_dbs,
        train_pred_sqls,
        train,
        os.path.join(args.predict_dir, "train_pred_results.json"),
        args.use_cache,
    )
    train_gold_results = _get_answers(
        llm,
        indexes,
        train_dbs,
        train_gold_sqls,
        train,
        os.path.join(args.predict_dir, "train_gold_results.json"),
        args.use_cache,
    )
    dev_pred_results = _get_answers(
        llm,
        indexes,
        dev_dbs,
        dev_pred_sqls,
        dev,
        os.path.join(args.predict_dir, "dev_pred_results.json"),
        args.use_cache,
    )
    dev_gold_results = _get_answers(
        llm,
        indexes,
        dev_dbs,
        dev_gold_sqls,
        dev,
        os.path.join(args.predict_dir, "dev_gold_results.json"),
        args.use_cache,
    )

    # Evaluate results.
    train_match = _match_answers(
        llm,
        train_gold_results,
        train_pred_results,
        train,
        os.path.join(args.predict_dir, "train_eval.json"),
    )
    print(f"Train match: {train_match:.4f}")
    dev_match = _match_answers(
        llm,
        dev_gold_results,
        dev_pred_results,
        dev,
        os.path.join(args.predict_dir, "dev_eval.json"),
    )
    print(f"Dev match: {dev_match:.4f}")
