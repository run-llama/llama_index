# LlamaIndex AgentMesh Integration

AgentMesh trust layer integration for LlamaIndex - enabling cryptographic identity verification and trust-gated agent workflows.

## Overview

This integration provides:

- **TrustedAgentWorker**: Agent worker with cryptographic identity and trust verification
- **TrustGatedQueryEngine**: Query engines with access control based on trust
- **Secure Data Access**: Governance layer for RAG pipelines with identity-based policies

## Installation

```bash
pip install llama-index-agent-agentmesh
```

## Quick Start

### Creating a Trusted Agent

```python
from llama_index.agent.agentmesh import TrustedAgentWorker, CMVKIdentity

# Generate cryptographic identity
identity = CMVKIdentity.generate(
    agent_name="research-agent",
    capabilities=["document_search", "summarization"],
)

# Create trusted agent worker
worker = TrustedAgentWorker.from_tools(
    tools=[search_tool, summarize_tool],
    identity=identity,
    llm=llm,
)

# Create agent with trust verification
agent = worker.as_agent()
```

### Trust-Gated Query Engine

```python
from llama_index.agent.agentmesh import TrustGatedQueryEngine, TrustPolicy

# Wrap query engine with trust policy
trusted_engine = TrustGatedQueryEngine(
    query_engine=base_engine,
    policy=TrustPolicy(
        min_trust_score=0.8,
        required_capabilities=["document_access"],
        audit_queries=True,
    ),
)

# Query requires verified identity
response = trusted_engine.query(
    "What are the quarterly results?",
    invoker_card=requester_card,
)
```

### Multi-Agent Trust Handoffs

```python
from llama_index.agent.agentmesh import TrustHandshake, TrustedAgentCard

# Create agent card for discovery
card = TrustedAgentCard(
    name="research-agent",
    description="Performs document research",
    capabilities=["search", "summarize"],
    identity=identity,
)
card.sign(identity)

# Verify peer before task handoff
handshake = TrustHandshake(my_identity=identity)
result = handshake.verify_peer(peer_card)

if result.trusted:
    # Safe to delegate task
    pass
```

## Features

### TrustedAgentWorker

An agent worker that:

- Has cryptographic identity for authentication
- Verifies peer agents before accepting tasks
- Signs outputs for verification by recipients
- Supports capability-based access control

### TrustGatedQueryEngine

A query engine wrapper that:

- Requires identity verification for queries
- Enforces trust score thresholds
- Restricts access based on capabilities
- Provides audit logging of all queries

### Data Access Governance

Control access to your RAG pipeline:

```python
from llama_index.agent.agentmesh import DataAccessPolicy

policy = DataAccessPolicy(
    allowed_collections=["public", "internal"],
    denied_collections=["confidential"],
    require_audit=True,
    max_results_per_query=100,
)

# Apply policy to index
trusted_index = TrustedVectorStoreIndex(
    index=base_index,
    policy=policy,
)
```

## Security Model

AgentMesh uses Ed25519 cryptography for:

- **Identity Generation**: Unique DID per agent
- **Request Signing**: All queries are signed
- **Response Verification**: Outputs can be verified

## API Reference

| Class                   | Description                          |
| ----------------------- | ------------------------------------ |
| `CMVKIdentity`          | Cryptographic agent identity         |
| `TrustedAgentWorker`    | Agent worker with trust verification |
| `TrustGatedQueryEngine` | Query engine with access control     |
| `TrustHandshake`        | Peer verification protocol           |
| `TrustedAgentCard`      | Agent discovery card                 |
| `DataAccessPolicy`      | RAG access governance                |

## License

MIT License
