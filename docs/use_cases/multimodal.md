# Multi-modal

LlamaIndex offers capabilities to not only build language-based applications, but also **multi-modal** applications - combining language and images.

## Types of Multi-modal Use Cases

This space is actively being explored right now, but there are some fascinating use cases popping up.

### Multi-Modal RAG

All the core RAG concepts: indexing, retrieval, and synthesis, can be extended into the image setting.

- The input could be text or image.
- The stored knowledge base can consist of text or images.
- The inputs to response generation can be text or image.
- The final response can be text or image.

Check out our guides below:

```{toctree}
---
maxdepth: 1
---
/examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb
Multi-modal retrieval with CLIP </examples/multi_modal/multi_modal_retrieval.ipynb>
```

### Retrieval-Augmented Image Captioning

Oftentimes understanding an image requires looking up information from a knowledge base. A flow here is retrieval-augmented image captioning - first caption the image with a multi-modal model, then refine the caption by retrieving from a text corpus.

Check out our guides below:

```{toctree}
---
maxdepth: 1
---
/examples/multi_modal/llava_multi_modal_tesla_10q.ipynb
```
