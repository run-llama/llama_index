from llama_index.embeddings.gaudi import GaudiEmbedding

if __name__ == "__main__":
    embed_model = GaudiEmbedding(
        embedding_input_size=-1,
        model_name="thenlper/gte-large",
    )

    # Basic embedding example
    embeddings = embed_model.get_text_embedding("It is raining cats and dogs here!")
    print(len(embeddings), embeddings[:10])
