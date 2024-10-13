from llama_index.embeddings.gaudi import GaudiHuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

if __name__ == "__main__":
    embed_model = LangchainEmbedding(
        GaudiHuggingFaceEmbeddings(
            embedding_input_size=-1,
            model_name="thenlper/gte-large",
            model_kwargs={"device": "hpu"},
        )
    )

    # Basic embedding example
    embeddings = embed_model.get_text_embedding("It is raining cats and dogs here!")
    print(len(embeddings), embeddings[:10])
