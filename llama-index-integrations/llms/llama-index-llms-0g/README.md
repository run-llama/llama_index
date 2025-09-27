# LlamaIndex LLMs Integration: 0G Compute Network

This package provides LlamaIndex integration for the 0G Compute Network, enabling decentralized AI inference with verification capabilities.

## Installation

```bash
pip install llama-index-llms-0g
```

## Prerequisites

The 0G Compute Network uses Ethereum-based authentication and requires:

1. **Ethereum Wallet**: You need an Ethereum private key for authentication
2. **0G Tokens**: Fund your account with OG tokens for inference payments
3. **Node.js Bridge** (Optional): For full JavaScript SDK integration

## Quick Start

### Basic Usage

```python
from llama_index.llms.zerog import ZeroGLLM

# Initialize with official model
llm = ZeroGLLM(
    model="llama-3.3-70b-instruct",  # or "deepseek-r1-70b"
    private_key="your_ethereum_private_key_here"
)

# Simple completion
response = llm.complete("Explain quantum computing in simple terms")
print(response.text)

# Chat interface
from llama_index.core.llms import ChatMessage, MessageRole

messages = [
    ChatMessage(role=MessageRole.USER, content="Hello, how are you?")
]
response = llm.chat(messages)
print(response.message.content)
```

### Streaming Responses

```python
# Streaming completion
for chunk in llm.stream_complete("Write a short story about AI"):
    print(chunk.delta, end="", flush=True)

# Streaming chat
messages = [
    ChatMessage(role=MessageRole.USER, content="Tell me about the 0G network")
]
for chunk in llm.stream_chat(messages):
    print(chunk.delta, end="", flush=True)
```

### Async Usage

```python
import asyncio

async def main():
    llm = ZeroGLLM(
        model="deepseek-r1-70b",
        private_key="your_private_key"
    )
    
    # Async completion
    response = await llm.acomplete("What is machine learning?")
    print(response.text)
    
    # Async streaming
    async for chunk in await llm.astream_complete("Explain neural networks"):
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

## Configuration Options

### Official Models

The integration supports two official 0G Compute Network models:

| Model | Provider Address | Description | Verification |
|-------|------------------|-------------|--------------|
| `llama-3.3-70b-instruct` | `0xf07240Efa67755B5311bc75784a061eDB47165Dd` | 70B parameter model for general AI tasks | TEE (TeeML) |
| `deepseek-r1-70b` | `0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3` | Advanced reasoning model | TEE (TeeML) |

### Custom Providers

```python
# Use a custom provider
llm = ZeroGLLM(
    model="custom-model-name",
    provider_address="0x1234567890abcdef...",
    private_key="your_private_key"
)
```

### Advanced Configuration

```python
llm = ZeroGLLM(
    model="llama-3.3-70b-instruct",
    private_key="your_private_key",
    rpc_url="https://evmrpc-testnet.0g.ai",  # or mainnet URL
    context_window=8192,
    max_tokens=1024,
    temperature=0.7,
    timeout=120.0,
    additional_kwargs={
        "top_p": 0.9,
        "frequency_penalty": 0.1
    }
)
```

## Account Management

### Funding Your Account

Before using the service, you need to fund your account with OG tokens:

```python
# Note: This requires the JavaScript SDK bridge (see Advanced Setup)
# For now, fund your account using the JavaScript SDK directly

# Example funding (requires JS bridge):
# await broker.ledger.addLedger("0.1")  # Add 0.1 OG tokens
```

### Checking Balance

```python
# This would require the JS bridge implementation
# await broker.ledger.getLedger()
```

## Advanced Setup (JavaScript SDK Bridge)

For full functionality including account management and verification, you'll need to set up a bridge to the JavaScript SDK.

### Option 1: Node.js Subprocess Bridge

Create a Node.js script that handles the 0G SDK operations:

```javascript
// 0g-bridge.js
const { ethers } = require("ethers");
const { createZGComputeNetworkBroker } = require("@0glabs/0g-serving-broker");

async function initializeBroker(privateKey, rpcUrl) {
    const provider = new ethers.JsonRpcProvider(rpcUrl);
    const wallet = new ethers.Wallet(privateKey, provider);
    return await createZGComputeNetworkBroker(wallet);
}

// Handle requests from Python
process.stdin.on('data', async (data) => {
    const request = JSON.parse(data.toString());
    // Handle different operations
    // Send response back to Python
});
```

### Option 2: HTTP Bridge Service

Create a simple HTTP service that wraps the JavaScript SDK:

```javascript
// 0g-service.js
const express = require('express');
const { ethers } = require("ethers");
const { createZGComputeNetworkBroker } = require("@0glabs/0g-serving-broker");

const app = express();
app.use(express.json());

app.post('/initialize', async (req, res) => {
    // Initialize broker
});

app.post('/inference', async (req, res) => {
    // Handle inference requests
});

app.listen(3000);
```

## Error Handling

```python
from llama_index.llms.zerog import ZeroGLLM

try:
    llm = ZeroGLLM(
        model="invalid-model",
        private_key="your_private_key"
    )
    response = llm.complete("Hello")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Runtime error: {e}")
```

## Integration with LlamaIndex

### With Query Engines

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.zerog import ZeroGLLM

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index with 0G LLM
llm = ZeroGLLM(
    model="llama-3.3-70b-instruct",
    private_key="your_private_key"
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)

# Query
response = query_engine.query("What is the main topic of these documents?")
print(response)
```

### With Chat Engines

```python
from llama_index.core import VectorStoreIndex
from llama_index.llms.zerog import ZeroGLLM

llm = ZeroGLLM(
    model="deepseek-r1-70b",
    private_key="your_private_key"
)

# Create chat engine
chat_engine = index.as_chat_engine(llm=llm)

# Chat
response = chat_engine.chat("Tell me about the documents")
print(response)
```

## Verification and Security

The 0G Compute Network provides verification capabilities:

- **TEE (Trusted Execution Environment)**: Official models run in verified environments
- **Cryptographic Proofs**: Responses can be cryptographically verified
- **Decentralized Infrastructure**: No single point of failure

## Troubleshooting

### Common Issues

1. **"Model not found" Error**
   ```python
   # Make sure you're using a valid model name
   llm = ZeroGLLM(model="llama-3.3-70b-instruct", ...)  # Correct
   # llm = ZeroGLLM(model="invalid-model", ...)  # Wrong
   ```

2. **Authentication Errors**
   ```python
   # Ensure your private key is valid and has sufficient funds
   # Check the RPC URL is correct for your network (testnet/mainnet)
   ```

3. **Timeout Issues**
   ```python
   # Increase timeout for longer requests
   llm = ZeroGLLM(timeout=300.0, ...)  # 5 minutes
   ```

### Getting Help

- **Documentation**: [0G Compute Network Docs](https://docs.0g.ai)
- **Discord**: Join the 0G community Discord
- **GitHub Issues**: Report bugs on the LlamaIndex repository

## Contributing

Contributions are welcome! Please see the main LlamaIndex contributing guidelines.

## License

This integration is licensed under the MIT License.

## Changelog

### v0.1.0

- Initial release
- Support for official 0G models (llama-3.3-70b-instruct, deepseek-r1-70b)
- Basic chat and completion interfaces
- Streaming support
- Async support
- Custom provider support
