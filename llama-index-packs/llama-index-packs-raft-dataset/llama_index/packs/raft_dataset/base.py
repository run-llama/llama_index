"""RAFT Dataset LlamaPack class."""

# Inspired from https://github.com/ShishirPatil/gorilla/tree/main/raft

from typing import Any, List
import random
import logging
import warnings

from datasets import Dataset

# Configure logging to output to the console, with messages of level DEBUG and above
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core import SimpleDirectoryReader

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

DEFAULT_CHUNK_SIZE = 512
DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD = 95


class RAFTDatasetPack(BaseLlamaPack):
    """RAFT Dataset Generator pack."""

    def __init__(
        self,
        file_path: str,
        llm: Any = None,
        embed_model: Any = None,
        num_questions_per_chunk: int = 5,
        num_distract_docs: int = 3,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        default_breakpoint_percentile_threshold=DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD,
    ):
        self.file_path = file_path
        self.num_questions_per_chunk = num_questions_per_chunk
        self.num_distract_docs = num_distract_docs
        self.chunk_size = chunk_size
        self.default_breakpoint_percentile_threshold = (
            default_breakpoint_percentile_threshold
        )
        self.ds = None
        self.llm = OpenAI(temperature=0, n=1, model="gpt-4") if llm is None else llm
        self.embed_model = OpenAIEmbedding() if embed_model is None else embed_model

    def strip_str(self, s) -> str:
        """
        Helper function for helping format strings returned by GPT-4.
        """
        if s.startswith("assistant:"):  # Check if the string starts with 'assistant '
            s = s.replace("assistant:", "", 1)  # Replace the first occurrence

        start_index, end_index = 0, len(s) - 1
        beg_found = False
        for i in range(len(s)):
            if s[i].isalpha():
                if not beg_found:
                    start_index = i
                    beg_found = True
                else:
                    end_index = i
        end_index += 2
        return s[start_index : min(end_index, len(s))]

    def encode_question_gen(self, question, chunk) -> List[str]:
        """
        Encode multiple prompt instructions into a single string for the general case.
        """
        prompt = f"""
            Question: {question}\nContext: {chunk}\n
            Answer this question using the information given in the context above. Here is things to pay attention to:
            - First provide step-by-step reasoning on how to answer the question.
            - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
            - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        """
        return [
            ChatMessage(
                role="system",
                content="You are a helpful question answerer who can provide an answer given a question and relevant context.",
            ),
            ChatMessage(role="user", content=prompt),
        ]

    def generate_label(self, question, context) -> str:
        """
        Generates the label / answer to `question` using `context` and GPT-4.
        """
        question_messages = self.encode_question_gen(question, context)
        response = self.llm.chat(question_messages)
        return str(response)

    def generate_instructions_gen(self, chunk, x=5) -> List[str]:
        """
        Generates `x` questions / use cases for `chunk`. Used when the input document is of general types
        `pdf`, `json`, or `txt`.
        """
        messages = [
            ChatMessage(
                role="system",
                content="You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'. The questions should be able to be answered in a few words or less."
                % (x),
            ),
            ChatMessage(role="user", content=str(chunk)),
        ]

        queries = str(self.llm.chat(messages)).split("\n")
        questions = [self.strip_str(q) for q in queries]
        questions = [q for q in questions if any(c.isalpha() for c in q)][:x]

        num_questions_generated = len(questions)
        if num_questions_generated < x:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({x})."
            )

        return questions

    def get_chunks(self, file_path: str, chunk_size: int) -> List[str]:
        """
        Takes in a `file_path`, retrieves the document, breaks it down into chunks of size
        `chunk_size`, and returns the chunks.
        """
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=self.default_breakpoint_percentile_threshold,
            embed_model=self.embed_model,
        )
        nodes = splitter.get_nodes_from_documents(documents)

        return [node.get_content() for node in nodes]

    def add_chunk_to_dataset(
        self,
        chunks: List,
        chunk: str,
        x: int = 5,
        num_distract: int = 3,
        p: float = 1.0,
    ):
        """
        Given a chunk, create {Q, A, D} triplets and add them to the dataset.
        """
        i = chunks.index(chunk)
        qs = self.generate_instructions_gen(chunk, x)
        for q in qs:
            datapt = {
                "id": None,
                "type": None,
                "question": None,
                "context": None,
                "oracle_context": None,
                "cot_answer": None,
            }

            datapt["id"] = f"seed_task_{0 if not self.ds else self.ds.num_rows}"
            datapt["type"] = "general"
            datapt["question"] = q

            # add distractor docs
            docs = [chunk]
            indices = list(range(len(chunks)))
            indices.remove(i)
            for j in random.sample(indices, num_distract):
                docs.append(chunks[j])
            # decides whether to add oracle document
            oracle = random.uniform(0, 1) < p
            if not oracle:
                docs[0] = chunks[random.sample(indices, 1)[0]]
            random.shuffle(docs)

            d = {"title": [], "sentences": []}

            d["title"].append(["placeholder_title"] * (num_distract + 1))
            d["sentences"].append(docs)
            datapt["context"] = d
            datapt["oracle_context"] = chunk

            # add answer to q
            datapt["cot_answer"] = self.generate_label(q, chunk)

            # construct model instruction
            context = ""
            for doc in docs:
                context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
            context += q
            datapt["instruction"] = context

            # add to dataset
            if not self.ds:
                # init ds
                datapt["id"] = [datapt["id"]]
                datapt["type"] = [datapt["type"]]
                datapt["question"] = [datapt["question"]]
                datapt["context"] = [datapt["context"]]
                datapt["oracle_context"] = [datapt["oracle_context"]]
                datapt["cot_answer"] = [datapt["cot_answer"]]
                datapt["instruction"] = [datapt["instruction"]]
                self.ds = Dataset.from_dict(datapt)
            else:
                self.ds = self.ds.add_item(datapt)

    def run(self) -> Any:
        """Run the pipeline."""
        chunks = self.get_chunks(self.file_path, self.chunk_size)

        logger.info(f"Number of chunks created: {len(chunks)}")

        self.num_distract_docs = (
            min(self.num_distract_docs, len(chunks)) - 1
        )  # should be less than number of chunks/ nodes created

        for index, chunk in enumerate(chunks):
            logger.info(f"Processing chunk: {index}")
            self.add_chunk_to_dataset(
                chunks, chunk, self.num_questions_per_chunk, self.num_distract_docs
            )

        return self.ds
