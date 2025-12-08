# LlamaIndex Embeddings Integration: Bedrock

This integration provides support for Amazon Bedrock embedding models through LlamaIndex.

## Installation

```bash
pip install llama-index-embeddings-bedrock
```

## Usage

```python
from llama_index.embeddings.bedrock import BedrockEmbedding

# Initialize the embedding model
embed_model = BedrockEmbedding(
    model_name="cohere.embed-english-v3",
    region_name="us-east-1",
)

# Get a single embedding
embedding = embed_model.get_text_embedding("Hello world")

# Get batch embeddings
embeddings = embed_model.get_text_embedding_batch(["Hello", "World"])
```

## Supported Models

### Amazon Titan

- `amazon.titan-embed-text-v1`
- `amazon.titan-embed-text-v2:0`
- `amazon.titan-embed-g1-text-02`

### Cohere

- `cohere.embed-english-v3`
- `cohere.embed-multilingual-v3`
- `cohere.embed-v4:0` (multimodal, supports text and images)

To list all supported models:

```python
from llama_index.embeddings.bedrock import BedrockEmbedding

supported_models = BedrockEmbedding.list_supported_models()
print(supported_models)
```

## Configuration

You can configure AWS credentials in several ways:

```python
# Option 1: Pass credentials directly
embed_model = BedrockEmbedding(
    model_name="cohere.embed-english-v3",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region_name="us-east-1",
)

# Option 2: Use AWS profile
embed_model = BedrockEmbedding(
    model_name="cohere.embed-english-v3",
    profile_name="your-aws-profile",
    region_name="us-east-1",
)

# Option 3: Use environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
embed_model = BedrockEmbedding(
    model_name="cohere.embed-english-v3",
)
```

## Cohere v4 Support

This integration supports both Cohere v3 and v4 embedding models, including the new multimodal `cohere.embed-v4:0` model. The integration automatically detects and handles different response formats (v3 and v4), maintaining full backward compatibility.

```python
# Using Cohere v4 model
embed_model = BedrockEmbedding(
    model_name="cohere.embed-v4:0",
    region_name="us-east-1",
)

# Text embeddings work seamlessly
embeddings = embed_model.get_text_embedding_batch(
    ["Hello world", "Another document"]
)
```

**Note:** Cohere v4 introduces a new response format that wraps embeddings in a `float` key when multiple embedding types are requested. This integration handles both the v3 format (`{"embeddings": [[...]]}`) and v4 formats (`{"embeddings": {"float": [[...]]}}` or `{"float": [[...]]}`) automatically.

## Use an Application Inference Profile

Amazon Bedrock supports user-created [Application Inference Profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html), which are [a sort of provisioned proxy to LLMs on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html) that allow for cost and model usage tracking.

Since these profile ARNs are account-specific, they must be handled specially in `BedrockEmbedding`.

When an application inference profile is created as an AWS resource, it references an existing Bedrock foundation model or a cross-region inference profile. The referenced model must be provided to the `BedrockEmbedding` initializer via the `model_name` argument, and the ARN of the application inference profile must be provided via the `application_inference_profile_arn` argument.

**Important:** `BedrockEmbedding` does _not_ validate that the `model_name` argument matches the underlying model referenced by the provided application inference profile. The caller is responsible for making sure that they match. As such, the behavior for when they _do not_ match is considered undefined.

```py
# Assumes the existence of a provisioned application inference profile
# that references a foundation model or cross-region inference profile.

from llama_index.embeddings.bedrock import BedrockEmbedding


# Instantiate the BedrockEmbedding model
# with the model_name and application_inference_profile
# Make sure the model is the one that the
# application inference profile refers to in AWS
embed_model = BedrockEmbedding(
    model_name="amazon.titan-embed-text-v2:0",  # this is the model referenced by the application inference profile
    application_inference_profile_arn="arn:aws:bedrock:us-east-1:012345678901:application-inference-profile/someProfileId",
)
```

## Examples

For more examples, see the [Bedrock Embeddings notebook](https://docs.llamaindex.ai/en/stable/examples/embeddings/bedrock/).
