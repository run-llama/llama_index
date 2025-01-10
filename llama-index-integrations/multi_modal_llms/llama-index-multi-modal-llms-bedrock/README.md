# LlamaIndex Multi-Modal LLM Integration: AWS Bedrock

AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models from leading AI companies through a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI.

## Installation

```bash
pip install llama-index-multi-modal-llms-bedrock
```

## Usage

Here's how to use the AWS Bedrock multi-modal integration:

### Basic Usage

```python
from llama_index.multi_modal_llms.bedrock import BedrockMultiModal
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import ImageDocument

# Initialize the model (credentials can be provided through environment variables)
llm = BedrockMultiModal(
    model="anthropic.claude-3-haiku-20240307-v1:0",  # or other Bedrock multi-modal models
    temperature=0.0,
    max_tokens=300,
    region_name="eu-central-1",  # make sure to use the region where the model access is granted
)

# Method 1: Load images using SimpleDirectoryReader
image_documents = SimpleDirectoryReader(
    input_files=["path/to/image.jpg"]
).load_data()

# Method 2: Create image documents directly
image_doc = ImageDocument(
    image_path="/path/to/image.jpg",  # Local file path
    # OR
    image="base64_encoded_image_string",  # Base64 encoded image
)

# Get a completion with both text and image
response = llm.complete(
    prompt="Describe this image in detail:",
    image_documents=image_documents,  # or [image_doc]
)

print(response.text)
```

### AWS Authentication

You can authenticate with AWS Bedrock in several ways:

1. Environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1  # optional
```

2. Explicit credentials:

```python
llm = BedrockMultiModal(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    aws_access_key_id="your_access_key",
    aws_secret_access_key="your_secret_key",
    region_name="eu-central-1",
)
```

3. AWS CLI configuration:

```bash
aws configure
```

4. IAM role-based authentication (when running on AWS services like EC2, Lambda, etc.)

### Supported Models

Currently supported multi-modal models in AWS Bedrock:

- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-3-5-haiku-20241022-v1:0`

### Advanced Usage

```python
# Using multiple images
image_docs = SimpleDirectoryReader(
    input_files=["image1.jpg", "image2.jpg"]
).load_data()

response = llm.complete(
    prompt="Compare these two images:", image_documents=image_docs
)

# Custom parameters
llm = BedrockMultiModal(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.0,
    max_tokens=300,
    timeout=60.0,  # API timeout in seconds
    max_retries=10,  # Maximum number of API retries
    additional_kwargs={
        # Add other model-specific parameters
    },
)

# Response includes token counts
print(f"Input tokens: {response.additional_kwargs['input_tokens']}")
print(f"Output tokens: {response.additional_kwargs['output_tokens']}")
```

## Development

To install development dependencies:

```bash
pip install -e ".[dev]"
```

To run tests:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License.
