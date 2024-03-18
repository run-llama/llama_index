"""Dataset Generator for Cross Encoder Finetuning."""

import re
from dataclasses import dataclass
from typing import List, Optional

from tqdm.auto import tqdm

from llama_index.legacy import VectorStoreIndex, get_tokenizer
from llama_index.legacy.llms import ChatMessage, OpenAI
from llama_index.legacy.llms.llm import LLM
from llama_index.legacy.node_parser import TokenTextSplitter
from llama_index.legacy.schema import Document, MetadataMode


@dataclass
class CrossEncoderFinetuningDatasetSample:
    """Class for keeping track of each item of Cross-Encoder training Dataset."""

    query: str
    context: str
    score: int


DEFAULT_QUERY_GEN_SYSTEM_PROMPT = """You are Albert a Professor proficient in {qa_topic}.
You are working on creating {num_questions_per_chunk} questions.
You provide the questions such that such that each separate is separated by a semicolon ';' so that different questions can be easily separated by the python split function"""


DEFAULT_QUERY_GEN_USER_PROMPT = """Take a deep breath, read through the below provided document and then create {num_questions_per_chunk} questions and respond with the created questions such that each separate question is separated by a semicolon ';' so that different questions can be easily separated by the python split function.
Document: {context}"""


def generate_synthetic_queries_over_documents(
    documents: List[Document],
    num_questions_per_chunk: int = 5,
    max_chunk_length: int = 3000,
    qa_topic: str = "everything",
    llm: Optional[LLM] = None,
    qa_generate_system_msg: str = DEFAULT_QUERY_GEN_SYSTEM_PROMPT,
    qa_generate_user_msg: str = DEFAULT_QUERY_GEN_USER_PROMPT,
) -> List[str]:
    questions = []
    node_parser = TokenTextSplitter(
        separator=" ",
        chunk_size=max_chunk_length,
        chunk_overlap=0,
        backup_separators=["\n"],
        tokenizer=get_tokenizer(),
    )

    llm = llm or OpenAI(model="gpt-3.5-turbo-16k", temperature=0.3)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    for node_id, text in tqdm(node_dict.items()):
        system_msg = qa_generate_system_msg.format(
            num_questions_per_chunk=num_questions_per_chunk, qa_topic=qa_topic
        )
        user_msg = qa_generate_user_msg.format(
            num_questions_per_chunk=num_questions_per_chunk, context=text
        )
        messages = [
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_msg),
        ]
        response = llm.chat(messages)
        response_content: str = (
            response.message.content if response.message.content is not None else ""
        )
        response_questions = re.split(";|\n", response_content)
        questions.extend(response_questions)

    return questions


# Query-Doc relevance prompt taken from OpenAI cookbook:-
# https://github.com/openai/openai-cookbook/blob/main/examples/Search_reranking_with_cross-encoders.ipynb
DEFAULT_QUERY_DOC_RELEVANCE_PROMPT = '''You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

Query: How to plant a tree?
Document: """Cars were invented in 1886, when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced horse-drawn carriages.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[7] The car is considered an essential part of the developed economy."""
Relevant: No

Query: Has the coronavirus vaccine been approved?
Document: """The Pfizer-BioNTech COVID-19 vaccine was approved for emergency use in the United States on December 11, 2020."""
Relevant: Yes

Query: What is the capital of France?
Document: """Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century, Gothic Notre-Dame cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré."""
Relevant: Yes

Query: What are some papers to learn about PPO reinforcement learning?
Document: """Proximal Policy Optimization and its Dynamic Version for Sequence Generation: In sequence generation task, many works use policy gradient for model optimization to tackle the intractable backpropagation issue when maximizing the non-differentiable evaluation metrics or fooling the discriminator in adversarial learning. In this paper, we replace policy gradient with proximal policy optimization (PPO), which is a proved more efficient reinforcement learning algorithm, and propose a dynamic approach for PPO (PPO-dynamic). We demonstrate the efficacy of PPO and PPO-dynamic on conditional sequence generation tasks including synthetic experiment and chit-chat chatbot. The results show that PPO and PPO-dynamic can beat policy gradient by stability and performance."""
Relevant: Yes

Query: Explain sentence embeddings
Document: """Inside the bubble: exploring the environments of reionisation-era Lyman-α emitting galaxies with JADES and FRESCO: We present a study of the environments of 16 Lyman-α emitting galaxies (LAEs) in the reionisation era (5.8<z<8) identified by JWST/NIRSpec as part of the JWST Advanced Deep Extragalactic Survey (JADES). Unless situated in sufficiently (re)ionised regions, Lyman-α emission from these galaxies would be strongly absorbed by neutral gas in the intergalactic medium (IGM). We conservatively estimate sizes of the ionised regions required to reconcile the relatively low Lyman-α velocity offsets (ΔvLyα<300kms−1) with moderately high Lyman-α escape fractions (fesc,Lyα>5%) observed in our sample of LAEs, indicating the presence of ionised ``bubbles'' with physical sizes of the order of 0.1pMpc≲Rion≲1pMpc in a patchy reionisation scenario where the bubbles are embedded in a fully neutral IGM. Around half of the LAEs in our sample are found to coincide with large-scale galaxy overdensities seen in FRESCO at z∼5.8-5.9 and z∼7.3, suggesting Lyman-α transmission is strongly enhanced in such overdense regions, and underlining the importance of LAEs as tracers of the first large-scale ionised bubbles. Considering only spectroscopically confirmed galaxies, we find our sample of UV-faint LAEs (MUV≳−20mag) and their direct neighbours are generally not able to produce the required ionised regions based on the Lyman-α transmission properties, suggesting lower-luminosity sources likely play an important role in carving out these bubbles. These observations demonstrate the combined power of JWST multi-object and slitless spectroscopy in acquiring a unique view of the early stages of Cosmic Reionisation via the most distant LAEs."""
Relevant: No

Query: {query}
Document: """{document}"""
Relevant:
'''


def generate_ce_fine_tuning_dataset(
    documents: List[Document],
    questions_list: List[str],
    max_chunk_length: int = 1000,
    llm: Optional[LLM] = None,
    qa_doc_relevance_prompt: str = DEFAULT_QUERY_DOC_RELEVANCE_PROMPT,
    top_k: int = 8,
) -> List[CrossEncoderFinetuningDatasetSample]:
    ce_dataset_list = []

    node_parser = TokenTextSplitter(
        separator=" ",
        chunk_size=max_chunk_length,
        chunk_overlap=0,
        backup_separators=["\n"],
        tokenizer=get_tokenizer(),
    )

    # Use logit bias in case of OpenAI for the tokens for Yes and No
    # to decrease the likelihood of any other tokens occurring
    llm = llm or OpenAI(
        model="gpt-3.5-turbo-16k", temperature=0.1, logit_bias={9642: 1, 2822: 1}
    )

    nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=top_k)

    for question in tqdm(questions_list):
        if question != "":
            retrieved_nodes = retriever.retrieve(question)
            for node in retrieved_nodes:
                node_content = node.get_text()
                msg_prompt = qa_doc_relevance_prompt.format(
                    query=question, document=node_content
                )
                response = llm.complete(msg_prompt)
                result = response.text.strip().lower()

                if result == "yes":
                    question_row = CrossEncoderFinetuningDatasetSample(
                        query=question, context=node_content, score=1
                    )
                    ce_dataset_list.append(question_row)
                elif result == "no":
                    question_row = CrossEncoderFinetuningDatasetSample(
                        query=question, context=node_content, score=0
                    )
                    ce_dataset_list.append(question_row)
                else:
                    pass

    return ce_dataset_list
