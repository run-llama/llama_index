# LlamaIndex Callbacks Integration: UpTrain

UpTrain ([github](https://github.com/uptrain-ai/uptrain) || [website](https://uptrain.ai/) || [docs](https://docs.uptrain.ai/getting-started/introduction)) is an open-source platform to evaluate and improve Generative AI applications. It provides grades for 20+ preconfigured checks (covering language, code, embedding use cases), performs root cause analysis on failure cases and gives insights on how to resolve them. Once you add UpTrainCallbackHandler to your existing LlamaIndex pipeline, it will automatically capture the right data, run evaluations and display the results in the output.

More details on UpTrain's evaluations can be found [here](https://github.com/uptrain-ai/uptrain?tab=readme-ov-file#pre-built-evaluations-we-offer-).

Selected operators from the LlamaIndex pipeline are highlighted for demonstration:

## 1. **RAG Query Engine Evaluations**:

The RAG query engine plays a crucial role in retrieving context and generating responses. To ensure its performance and response quality, we conduct the following evaluations:

- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the context extracted from the query is relevant to the response.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM is hallucinating or providing incorrect information.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information requested by the query.

## 2. **Sub-Question Query Generation Evaluation**:

The SubQuestionQueryGeneration operator decomposes a question into sub-questions, generating responses for each using a RAG query engine. To evaluate the performance of SubQuery module, we add another check as well as run the above three for all the sub-queries:

- **[Sub Query Completeness](https://docs.uptrain.ai/predefined-evaluations/query-quality/sub-query-completeness)**: Assures that the sub-questions accurately and comprehensively cover the original query.

## 3. **Re-Ranking Evaluations**:

Re-ranking involves reordering nodes based on relevance to the query and choosing the top n nodes. Different evaluations are performed based on the number of nodes returned after re-ranking.

a. Same Number of Nodes

- **[Context Reranking](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-reranking)**: Checks if the order of re-ranked nodes is more relevant to the query than the original order.

b. Different Number of Nodes:

- **[Context Conciseness](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-conciseness)**: Examines whether the reduced number of nodes still provides all the required information.

These evaluations collectively ensure the robustness and effectiveness of the RAG query engine, SubQuestionQueryGeneration operator, and the re-ranking process in the LlamaIndex pipeline.
