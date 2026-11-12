"""
Pydantic models for A2A protocol types.

Defines the core data structures used in the A2A (Agent-to-Agent) protocol:
- AgentCard: Discovery document describing agent capabilities
- Skill: Individual capability definition
- Task: Unit of work exchanged between agents
- Message: Communication unit with parts (text, file, etc.)
- Part: Content fragment within a message
- TaskResult: Response from skill/task execution
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl


class Part(BaseModel):
    """A single content part within a message or task input/output."""
    kind: Literal["text", "file", "image"]
    text: Optional[str] = None
    file_uri: Optional[str] = None
    mime_type: Optional[str] = None


class Message(BaseModel):
    """A message exchanged between agents."""
    role: Literal["user", "assistant", "system"]
    parts: List[Part]
    message_id: Optional[str] = None
    context_id: Optional[str] = None


class TaskInput(BaseModel):
    """Input to a task."""
    parts: List[Part]


class TaskOutput(BaseModel):
    """Output from a task."""
    parts: List[Part]
    status: Optional[str] = None


class Task(BaseModel):
    """A task represents a unit of work to be performed by an agent."""
    id: str
    name: str
    input: TaskInput
    output: Optional[TaskOutput] = None
    status: str = "submitted"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Capability(BaseModel):
    """Describes a specific capability an agent offers."""
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = Field(default_factory=list)


class Skill(Capability):
    """A named capability that can be invoked by other agents."""
    name: str
    endpoint: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentCapabilities(BaseModel):
    """Agent capabilities as defined by A2A protocol."""
    streaming: bool = False
    push_notifications: bool = False


class AgentProvider(BaseModel):
    """Agent provider information."""
    organization: str
    url: str


class AgentCard(BaseModel):
    """
    The AgentCard is the discovery document for an A2A agent.
    It describes the agent's identity, capabilities, and endpoints.

    Typically served at /.well-known/agent-card.json
    """
    url: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    skills: List[Skill] = Field(default_factory=list)
    capabilities: Optional[AgentCapabilities] = None
    provider: Optional[AgentProvider] = None
    authentication: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    default_input_modes: Optional[List[str]] = None
    default_output_modes: Optional[List[str]] = None
    preferred_transport: Optional[str] = None
    protocol_version: Optional[str] = None


class TaskResult(BaseModel):
    """Result of a task/skill execution."""
    task_id: str
    output: Optional[TaskOutput] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    """Standard JSON-RPC 2.0 response."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[int] = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: Optional[Any] = None
