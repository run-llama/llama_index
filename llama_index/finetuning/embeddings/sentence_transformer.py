"""Sentence Transformer Finetuning Engine."""

from llama_index.embeddings.base import BaseEmbedding

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from llama_index.schema import TextNode, MetadataMode
from llama_index.llms.openai import OpenAI
from llama_index.llms.base import LLM
from llama_index.embeddings.utils import resolve_embed_model
from tqdm import tqdm
import uuid
import re


class EmbeddingQAFinetuneDataset(BaseModel):
    """Embedding QA Finetuning Dataset."""

    queries: Dict[str, str]  # dict id -> query
    corpus: Dict[str, str]  # dict id -> string
    relevant_docs: Dict[str, List[str]]  # query id -> list of doc ids


DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


# generate queries as a convenience function
def generate_qa_embedding_pairs(
    nodes: List[TextNode],
    llm: Optional[LLM] = None,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk=2,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    llm = OpenAI(model="gpt-3.5-turbo")

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(node_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]
    return queries, relevant_docs


class SentenceTransformersFinetuningEngine:
    """Sentence Transformers Finetuning Engine."""

    def __init__(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        model_id: str = "BAAI/bge-small-en",
        model_output_path: str = "exp_finetune",
        batch_size: int = 10,
        val_dataset: Optional[EmbeddingQAFinetuneDataset] = None,
    ) -> None:
        """Init params."""
        from sentence_transformers import InputExample, SentenceTransformer
        from torch.utils.data import DataLoader

        self.dataset = dataset

        self.model_id = model_id
        self.model_output_path = model_output_path
        self.model = SentenceTransformer(model_id)

        # TODO: support more than 1 doc per query
        examples = []
        for query_id, query in dataset.queries.items():
            node_id = dataset.relevant_docs[query_id][0]
            text = dataset.corpus[node_id]
            example = InputExample(texts=[query, text])
            examples.append(example)
        self.examples = examples

        self.loader = DataLoader(examples, batch_size=batch_size)

        # define evaluator
        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        evaluator: Optional[InformationRetrievalEvaluator] = None
        if val_dataset is not None:
            evaluator = InformationRetrievalEvaluator(
                val_dataset.queries, val_dataset.corpus, val_dataset.relevant_docs
            )
        self.evaluator = evaluator

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        from sentence_transformers import losses

        loss = losses.MultipleNegativesRankingLoss(self.model)
        epochs = train_kwargs.get("epochs", 2)
        warmup_steps = int(len(self.loader) * epochs * 0.1)
        output_path = train_kwargs.get("output_path", "exp_finetune")
        show_progress_bar = train_kwargs.get("show_progress_bar", True)
        evaluation_steps = train_kwargs.get("evaluation_steps", 50)

        self.model.fit(
            train_objectives=[(self.loader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=evaluation_steps,
        )

    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Gets finetuned model."""

        embed_model_str = "local:" + self.model_output_path
        embed_model = resolve_embed_model(embed_model_str)

        return embed_model
