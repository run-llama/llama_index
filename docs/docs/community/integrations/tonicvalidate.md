# Tonic Validate

## What is Tonic Validate

Tonic Validate is a tool for people developing retrieval augmented generation (RAG) systems to evaluate the performance of their system. You can use Tonic Validate for one-off spot checks of your LlamaIndex setup's performance or you can even use it inside an existing CI/CD system like Github Actions. There are two parts to Tonic Validate

1. The Open-Source SDK
2. [The Web UI](https://validate.tonic.ai/)

You can use the SDK without using the Web UI if you prefer. The SDK includes all of the tools needed to evaluate your RAG system. The purpose of the web UI is to provide a layer on top of the SDK for visualizing your results. This allows you to get a better sense of your system's performance as opposed to viewing only raw numbers.

If you want to use the web UI you can go to [here](https://validate.tonic.ai/) to sign up for a free account.

## How to use Tonic Validate

### Setting Up Tonic Validate

You can install Tonic Validate via the following command

```
pip install tonic-validate
```

To use Tonic Validate, you need to provide an OpenAI key as the score calculations use an LLM on the backend. You can set an OpenAI key via setting the `OPENAI_API_KEY` environmental variable to your OpenAI API key.

```python
import os

os.environ["OPENAI_API_KEY"] = "put-your-openai-api-key-here"
```

If you are uploading your results to the UI, also make sure to set your Tonic Validate API key you received during the account set up for the [web UI](https://validate.tonic.ai/). If you have not already set up your account on the web UI you can do so [here](https://validate.tonic.ai/). Once you have the API Key you can set it via the `TONIC_VALIDATE_API_KEY` environment variable.

```python
import os

os.environ["TONIC_VALIDATE_API_KEY"] = "put-your-validate-api-key-here"
```

### One Question Usage Example

For this example, we have an example of a question with a reference correct answer that does not match the LLM response answer. There are two retrieved context chunks, of which one of them has the correct answer.

```python
question = "What makes Sam Altman a good founder?"
reference_answer = "He is smart and has a great force of will."
llm_answer = "He is a good founder because he is smart."
retrieved_context_list = [
    "Sam Altman is a good founder. He is very smart.",
    "What makes Sam Altman such a good founder is his great force of will.",
]
```

The answer similarity score is a score between 0 and 5 that scores how well the LLM answer matches the reference answer. In this case, they do not match perfectly, so the answer similarity score is not a perfect 5.

```python
answer_similarity_evaluator = AnswerSimilarityEvaluator()
score = await answer_similarity_evaluator.aevaluate(
    question,
    llm_answer,
    retrieved_context_list,
    reference_response=reference_answer,
)
print(score)
# >> EvaluationResult(query='What makes Sam Altman a good founder?', contexts=['Sam Altman is a good founder. He is very smart.', 'What makes Sam Altman such a good founder is his great force of will.'], response='He is a good founder because he is smart.', passing=None, feedback=None, score=4.0, pairwise_source=None, invalid_result=False, invalid_reason=None)
```

The answer consistency score is between 0.0 and 1.0, and measure whether the answer has information that does not appear in the retrieved context. In this case, the answer does appear in the retrieved context, so the score is 1.

```python
answer_consistency_evaluator = AnswerConsistencyEvaluator()


score = await answer_consistency_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
print(score)
# >> EvaluationResult(query='What makes Sam Altman a good founder?', contexts=['Sam Altman is a good founder. He is very smart.', 'What makes Sam Altman such a good founder is his great force of will.'], response='He is a good founder because he is smart.', passing=None, feedback=None, score=1.0, pairwise_source=None, invalid_result=False, invalid_reason=None)
```

Augmentation accuracy measures the percentage of the retrieved context that is in the answer. In this case, one of the retrieved contexts is in the answer, so this score is 0.5.

```python
augmentation_accuracy_evaluator = AugmentationAccuracyEvaluator()


score = await augmentation_accuracy_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
print(score)
# >> EvaluationResult(query='What makes Sam Altman a good founder?', contexts=['Sam Altman is a good founder. He is very smart.', 'What makes Sam Altman such a good founder is his great force of will.'], response='He is a good founder because he is smart.', passing=None, feedback=None, score=0.5, pairwise_source=None, invalid_result=False, invalid_reason=None)
```

Augmentation precision measures whether the relevant retrieved context makes it into the answer. Both of the retrieved contexts are relevant, but only one makes it into the answer. For that reason, this score is 0.5.

```python
augmentation_precision_evaluator = AugmentationPrecisionEvaluator()


score = await augmentation_precision_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
print(score)
# >> EvaluationResult(query='What makes Sam Altman a good founder?', contexts=['Sam Altman is a good founder. He is very smart.', 'What makes Sam Altman such a good founder is his great force of will.'], response='He is a good founder because he is smart.', passing=None, feedback=None, score=0.5, pairwise_source=None, invalid_result=False, invalid_reason=None)
```

Retrieval precision measures the percentage of retrieved context is relevant to answer the question. In this case, both of the retrieved contexts are relevant to answer the question, so the score is 1.0.

```python
retrieval_precision_evaluator = RetrievalPrecisionEvaluator()


score = await retrieval_precision_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
print(score)
# >> EvaluationResult(query='What makes Sam Altman a good founder?', contexts=['Sam Altman is a good founder. He is very smart.', 'What makes Sam Altman such a good founder is his great force of will.'], response='He is a good founder because he is smart.', passing=None, feedback=None, score=1.0, pairwise_source=None, invalid_result=False, invalid_reason=None)
```

The TonicValidateEvaluator can calculate all of Tonic Validate's metrics at once.

```python
tonic_validate_evaluator = TonicValidateEvaluator()


scores = await tonic_validate_evaluator.aevaluate(
    question,
    llm_answer,
    retrieved_context_list,
    reference_response=reference_answer,
)
print(scores.score_dict)
# >> {
#     'answer_consistency': 1.0,
#     'answer_similarity': 4.0,
#     'augmentation_accuracy': 0.5,
#     'augmentation_precision': 0.5,
#     'retrieval_precision': 1.0
# }
```

### Evaluating multiple questions at once

You can also evaluate more than one query and response at once using TonicValidateEvaluator, and return a tonic_validate Run object that can be logged to the [Tonic Validate UI](https://validate.tonic.ai).

To do this, you put the questions, LLM answers, retrieved context lists, and reference answers into lists and call evaluate_run.

```python
questions = ["What is the capital of France?", "What is the capital of Spain?"]
reference_answers = ["Paris", "Madrid"]
llm_answer = ["Paris", "Madrid"]
retrieved_context_lists = [
    [
        "Paris is the capital and most populous city of France.",
        "Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture.",
    ],
    [
        "Madrid is the capital and largest city of Spain.",
        "Madrid, Spain's central capital, is a city of elegant boulevards and expansive, manicured parks such as the Buen Retiro.",
    ],
]


tonic_validate_evaluator = TonicValidateEvaluator()


scores = await tonic_validate_evaluator.aevaluate_run(
    [questions], [llm_answers], [retrieved_context_lists], [reference_answers]
)
print(scores.run_data[0].scores)
# >> {
#     'answer_consistency': 1.0,
#     'answer_similarity': 3.0,
#     'augmentation_accuracy': 0.5,
#     'augmentation_precision': 0.5,
#     'retrieval_precision': 1.0
# }
```

### Uploading Results to the UI

If you want to upload your scores to the UI, then you can use the Tonic Validate API. Before doing so, make sure you have `TONIC_VALIDATE_API_KEY` set as described in the [Setting Up Tonic Validate](#setting-up-tonic-validate) section. You also need to make sure you have a project created in the Tonic Validate UI and that you have copied the project id. After the API Key and project are set up, you can initialize the Validate API and upload the results.

```python
validate_api = ValidateApi()
project_id = "your-project-id"
validate_api.upload_run(project_id, scores)
```

Now you can see your results in the Tonic Validate UI!

![Tonic Validate Graph](../../_static/integrations/TonicValidate-Graph.png)

### End to End Example

Here we will show you how to use Tonic Validate End To End with Llama Index. First, let's download a dataset for Llama Index to run on using the Llama Index CLI.

```bash
llamaindex-cli download-llamadataset EvaluatingLlmSurveyPaperDataset --download-dir ./data
```

Now, we can create a python file called `llama.py` and put the following code in it.

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex


documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()
```

This code essentially just loads in the dataset files and then initializes Llama Index.

Llama Index's CLI also downloads a list of questions and answers you can use for testing on their example dataset. If you want to use these questions and answers, you can use the code below.

```python
from llama_index.core.llama_dataset import LabelledRagDataset

rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")


# We are only going to do 10 questions as running through the full data set takes too long
questions = [item.query for item in rag_dataset.examples][:10]
reference_answers = [item.reference_answer for item in rag_dataset.examples][
    :10
]
```

Now we can query for the responses from Llama Index.

```python
llm_answers = []
retrieved_context_lists = []
for question in questions:
    response = query_engine.query(question)
    context_list = [x.text for x in response.source_nodes]
    retrieved_context_lists.append(context_list)
    llm_answers.append(response.response)
```

Now to score it, we can do the following

```
from tonic_validate.metrics import AnswerSimilarityMetric
from llama_index.evaluation.tonic_validate import TonicValidateEvaluator


tonic_validate_evaluator = TonicValidateEvaluator(
    metrics=[AnswerSimilarityMetric()], model_evaluator="gpt-4-1106-preview"
)

scores = tonic_validate_evaluator.evaluate_run(
    questions, retrieved_context_lists, reference_answers, llm_answers
)
print(scores.overall_scores)
```

If you want to upload your scores to the UI, then you can use the Tonic Validate API. Before doing so, make sure you have `TONIC_VALIDATE_API_KEY` set as described in the [Setting Up Tonic Validate](#setting-up-tonic-validate) section. You also need to make sure you have a project created in the Tonic Validate UI and that you have copied the project id. After the API Key and project are set up, you can initialize the Validate API and upload the results.

```python
validate_api = ValidateApi()
project_id = "your-project-id"
validate_api.upload_run(project_id, run)
```

## More Documentation

In addition to the documentation here, you can also visit [Tonic Validate's Github page](https://github.com/TonicAI/tonic_validate) for more documentation about how to interact with our API for uploading results.
