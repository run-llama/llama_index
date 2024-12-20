# Embeddings

##### FAQ

1. [How to use a custom/local embedding model?](#1-how-to-use-a-customlocal-embedding-model)
2. [How to use a local hugging face embedding model?](#2-how-to-use-a-local-hugging-face-embedding-model)
3. [How to use embedding model to generate embeddings for text?](#3-how-to-use-embedding-model-to-generate-embeddings-for-text)
4. [How to use Huggingface Text-Embedding Inference with LlamaIndex?](#4-how-to-use-huggingface-text-embedding-inference-with-llamaindex)

---

##### 1. How to use a custom/local embedding model?

To create your customized embedding class you can follow [Custom Embeddings](../../examples/embeddings/custom_embeddings.ipynb) guide.

---

##### 2. How to use a local hugging face embedding model?

To use a local HuggingFace embedding model you can follow [Local Embeddings with HuggingFace](../../examples/embeddings/huggingface.ipynb) guide.

---

##### 3. How to use embedding model to generate embeddings for text?

You can generate embeddings for texts with the following piece of code.

```py
text_embedding = embed_model.get_text_embedding("YOUR_TEXT")
```

---

##### 4. How to use Huggingface Text-Embedding Inference with LlamaIndex?

To use HuggingFace Text-Embedding Inference you can follow [Text-Embedding-Inference](../../examples/embeddings/text_embedding_inference.ipynb) tutorial.
