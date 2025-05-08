"""Helper functions for Titanic GPT-3 experiments."""

# form prompt, run GPT
import re
from typing import List, Optional, Tuple

import pandas as pd
from llama_index.indices.utils import extract_numbers_given_response
from llama_index.llms import OpenAI
from llama_index.prompts import BasePromptTemplate, PromptTemplate
from sklearn.model_selection import train_test_split


def get_train_and_eval_data(
    csv_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Get train and eval data."""
    df = pd.read_csv(csv_path)
    label_col = "Survived"
    cols_to_drop = ["PassengerId", "Ticket", "Name", "Cabin"]
    df = df.drop(cols_to_drop, axis=1)
    labels = df.pop(label_col)
    train_df, eval_df, train_labels, eval_labels = train_test_split(
        df, labels, test_size=0.25, random_state=0
    )
    return train_df, train_labels, eval_df, eval_labels


def get_sorted_dict_str(d: dict) -> str:
    """Get sorted dict string."""
    keys = sorted(d.keys())
    return "\n".join([f"{k}:{d[k]}" for k in keys])


def get_label_str(labels: pd.Series, i: int) -> str:
    """Get label string."""
    return f"{labels.name}: {labels.iloc[i]}"


def get_train_str(
    train_df: pd.DataFrame, train_labels: pd.Series, train_n: int = 10
) -> str:
    """Get train str."""
    dict_list = train_df.to_dict("records")[:train_n]
    item_list = []
    for i, d in enumerate(dict_list):
        dict_str = get_sorted_dict_str(d)
        label_str = get_label_str(train_labels, i)
        item_str = (
            f"This is the Data:\n{dict_str}\nThis is the correct answer:\n{label_str}"
        )
        item_list.append(item_str)

    return "\n\n".join(item_list)


def extract_float_given_response(response: str, n: int = 1) -> Optional[float]:
    """
    Extract number given the GPT-generated response.

    Used by tree-structured indices.

    """
    numbers = re.findall(r"\d+\.\d+", response)
    if len(numbers) == 0:
        # if no floats, try extracting ints, and convert to float
        new_numbers = extract_numbers_given_response(response, n=n)
        if new_numbers is None:
            return None
        else:
            return float(new_numbers[0])
    else:
        return float(numbers[0])


def get_eval_preds(
    train_prompt: BasePromptTemplate, train_str: str, eval_df: pd.DataFrame, n: int = 20
) -> List:
    """Get eval preds."""
    llm = OpenAI()
    eval_preds = []
    for i in range(n):
        eval_str = get_sorted_dict_str(eval_df.iloc[i].to_dict())
        response = llm.predict(train_prompt, train_str=train_str, eval_str=eval_str)
        pred = extract_float_given_response(response)
        print(f"Getting preds: {i}/{n}: {pred}")
        if pred is None:
            # something went wrong, impute a 0.5
            eval_preds.append(0.5)
        else:
            eval_preds.append(pred)
    return eval_preds


# default train prompt

train_prompt_str = (
    "The following structured data is provided in "
    '"Feature Name":"Feature Value" format.\n'
    "Each datapoint describes a passenger on the Titanic.\n"
    "The task is to decide whether the passenger survived.\n"
    "Some example datapoints are given below: \n"
    "-------------------\n"
    "{train_str}\n"
    "-------------------\n"
    "Given this, predict whether the following passenger survived. "
    "Return answer as a number between 0 or 1. \n"
    "{eval_str}\n"
    "Survived: "
)

train_prompt = PromptTemplate(template=train_prompt_str)


# prompt to summarize the data
query_str = "Which is the relationship between these features and predicting survival?"
qa_data_str = (
    "The following structured data is provided in "
    '"Feature Name":"Feature Value" format.\n'
    "Each datapoint describes a passenger on the Titanic.\n"
    "The task is to decide whether the passenger survived.\n"
    "Some example datapoints are given below: \n"
    "-------------------\n"
    "{context_str}\n"
    "-------------------\n"
    "Given this, answer the question: {query_str}"
)

qa_data_prompt = PromptTemplate(template=qa_data_str)

# prompt to refine the answer
refine_str = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "The following structured data is provided in "
    '"Feature Name":"Feature Value" format.\n'
    "Each datapoint describes a passenger on the Titanic.\n"
    "The task is to decide whether the passenger survived.\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more datapoints below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
refine_prompt = PromptTemplate(template=refine_str)


# train prompt with refined context

train_prompt_with_context_str = (
    "The following structured data is provided in "
    '"Feature Name":"Feature Value" format.\n'
    "Each datapoint describes a passenger on the Titanic.\n"
    "The task is to decide whether the passenger survived.\n"
    "We discovered the following relationship between features and survival:\n"
    "-------------------\n"
    "{train_str}\n"
    "-------------------\n"
    "Given this, predict whether the following passenger survived. \n"
    "Return answer as a number between 0 or 1. \n"
    "{eval_str}\n"
    "Survived: "
)

train_prompt_with_context = PromptTemplate(template=train_prompt_with_context_str)
