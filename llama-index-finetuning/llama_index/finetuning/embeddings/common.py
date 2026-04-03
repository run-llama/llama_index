import json
import re
import uuid
import warnings
from typing import Dict, List, Tuple
from tqdm import tqdm
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.utils import LLM
from llama_index.core.schema import MetadataMode, TextNode


class EmbeddingQAFinetuneDataset(BaseModel):
    """
    Embedding QA Finetuning Dataset.

    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.

    """

    queries: Dict[str, str]
    corpus: Dict[str, str]
    relevant_docs: Dict[str, List[str]]
    mode: str = "text"

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """
        Save the dataset to a JSON file.

        Args:
            path (str): The file path to save the JSON.

        """
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "EmbeddingQAFinetuneDataset":
        """
        Load the dataset from a JSON file.

        Args:
            path (str): The file path to load the JSON from.

        Returns:
            EmbeddingQAFinetuneDataset: The loaded dataset.

        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


def load_existing_data(
    path: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Load existing data from a JSON file if it exists.

    Args:
        path (str): The file path to load the JSON from.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]: The loaded queries, corpus, and relevant_docs.

    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data["queries"], data["corpus"], data["relevant_docs"]
    except FileNotFoundError:
        return {}, {}, {}


def generate_qa_embedding_pairs(
    nodes: List[TextNode],
    llm: LLM,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
    retry_limit: int = 3,
    on_failure: str = "continue",  # options are "fail" or "continue"
    save_every: int = 500,
    output_path: str = "qa_finetune_dataset.json",
    verbose: bool = True,
) -> EmbeddingQAFinetuneDataset:
    """
    Generate QA pairs from a set of nodes and save periodically.

    Args:
        nodes (List[TextNode]): List of TextNode objects to process.
        llm (LLM): The large language model to use for generating questions.
        qa_generate_prompt_tmpl (str): The template for generating QA prompts.
        num_questions_per_chunk (int): Number of questions to generate per chunk of text.
        retry_limit (int): Number of times to retry on failure.
        on_failure (str): Action to take on repeated failures ('fail' or 'continue').
        save_every (int): Number of nodes to process before saving the dataset.
        output_path (str): The file path to save the JSON output.
        verbose (bool): If True, print debugging messages.

    Returns:
        EmbeddingQAFinetuneDataset: The generated dataset.

    """
    queries, corpus, relevant_docs = load_existing_data(output_path)

    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    start_index = len(corpus)

    save_counter = start_index

    for node_id, text in tqdm(
        list(node_dict.items())[start_index:], initial=start_index
    ):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )

        retry_count = 0
        success = False
        while retry_count < retry_limit:
            try:
                response = llm.complete(query)
                success = True
                break
            except Exception as e:
                retry_count += 1
                if verbose:
                    print(
                        f"Error querying LLM: {e}. Retrying {retry_count}/{retry_limit}..."
                    )

        if not success:
            if on_failure == "fail":
                raise RuntimeError(f"Failed to query LLM after {retry_limit} retries.")
            elif on_failure == "continue":
                if verbose:
                    print(f"Skipping node {node_id} after {retry_limit} retries.")
                continue

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0][
            :num_questions_per_chunk
        ]

        num_questions_generated = len(questions)
        if num_questions_generated < num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({num_questions_per_chunk})."
            )

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

        corpus[node_id] = text

        save_counter += 1
        if save_counter % save_every == 0:
            dataset = EmbeddingQAFinetuneDataset(
                queries=queries, corpus=corpus, relevant_docs=relevant_docs
            )
            dataset.save_json(output_path)
            if verbose:
                print(f"Saved progress at {save_counter} entries.")

    # Save final dataset
    dataset = EmbeddingQAFinetuneDataset(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs
    )
    dataset.save_json(output_path)
    if verbose:
        print("Final dataset saved.")

    return dataset
