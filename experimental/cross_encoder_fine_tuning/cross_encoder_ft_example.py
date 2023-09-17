import json
import os
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
import openai
from llama_index.finetuning.cross_encoders.dataset_gen import generate_ce_fine_tuning_dataset, \
    generate_synthetic_queries_over_documents

from llama_index.finetuning.cross_encoders.cross_encoder import CrossEncoderFinetuneEngine

os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = os.environ["OPENAI_API_KEY"]


def main():
    train_files = ["docs/examples/data/10k/lyft_2021.pdf"]

    # Load the train files
    reader = SimpleDirectoryReader(input_files=train_files)
    docs = reader.load_data()

    # Generate synthetic questions over the train files
    questions = generate_synthetic_queries_over_documents(documents=docs, max_chunk_length=5000,
                                                          qa_topic='Business and Finance', num_questions_per_chunk=1)

    # Generate fine-tuning dataset for cross-encoder using the generated synthetic questions
    finetuning_dataset = generate_ce_fine_tuning_dataset(documents=docs, questions_list=questions,
                                                         max_chunk_length=2000)

    # Initialise the cross-encoder fine-tuning engine
    finetuning_engine = CrossEncoderFinetuneEngine(dataset=finetuning_dataset)

    # Finetune the cross-encoder model
    finetuning_engine.finetune()

    # get the cross-encoder fine-tuning model
    cross_encoder_model = finetuning_engine.get_finetuned_model()


if __name__ == '__main__':
    main()





