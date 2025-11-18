"""
FastAPI server with LlamaIndex integration
Production-ready example with health checks, error handling, and streaming
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import logging
from datetime import datetime
from pathlib import Path
import asyncio

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Optional: Vector store imports (uncomment as needed)
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from llama_index.vector_stores.qdrant import QdrantVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex API Server",
    description="Production-ready LlamaIndex API with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

# Global variables
query_engine = None
chat_engine = None
index = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(5, description="Number of relevant chunks to retrieve")
    stream: Optional[bool] = Field(False, description="Stream the response")

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    stream: Optional[bool] = Field(False, description="Stream the response")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    files: Optional[List[str]] = Field(None, description="List of file paths to ingest")
    rebuild: Optional[bool] = Field(False, description="Rebuild entire index")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    index_loaded: bool


def configure_llm_and_embeddings():
    """Configure LLM and embedding models"""
    try:
        # Configure LLM
        Settings.llm = OpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            api_key=OPENAI_API_KEY,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512"))
        )

        # Configure embeddings
        Settings.embed_model = OpenAIEmbedding(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=OPENAI_API_KEY
        )

        # Configure node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "20"))
        )

        logger.info("LLM and embeddings configured successfully")
    except Exception as e:
        logger.error(f"Error configuring LLM/embeddings: {e}")
        raise


def load_or_create_index():
    """Load existing index or create new one"""
    global index, query_engine, chat_engine

    try:
        if STORAGE_DIR.exists() and list(STORAGE_DIR.glob("*")):
            logger.info("Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded successfully")
        else:
            logger.info("Creating new index...")

            # Check if data directory has files
            if not list(DATA_DIR.glob("*")):
                logger.warning("No documents found in data directory")
                return None

            # Load documents
            documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
            logger.info(f"Loaded {len(documents)} documents")

            # Create index
            index = VectorStoreIndex.from_documents(documents)

            # Persist index
            index.storage_context.persist(persist_dir=str(STORAGE_DIR))
            logger.info("Index created and persisted successfully")

        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
            streaming=True
        )

        # Create chat engine
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
            streaming=True
        )

        return index

    except Exception as e:
        logger.error(f"Error loading/creating index: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting LlamaIndex API Server...")

    # Check for API key
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Configure LLM and embeddings
    configure_llm_and_embeddings()

    # Load or create index
    try:
        load_or_create_index()
    except Exception as e:
        logger.warning(f"Could not load index on startup: {e}")
        logger.info("Index will be created when documents are ingested")

    logger.info("Server started successfully")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "LlamaIndex API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        environment=ENVIRONMENT,
        index_loaded=index is not None
    )


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document index

    Returns relevant information from indexed documents
    """
    if query_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Query engine not initialized. Please ingest documents first."
        )

    try:
        logger.info(f"Processing query: {request.query}")

        # Configure query engine with custom top_k if provided
        custom_query_engine = index.as_query_engine(
            similarity_top_k=request.top_k,
            streaming=request.stream
        )

        # Execute query
        response = custom_query_engine.query(request.query)

        # Extract source information
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    "text": node.node.text[:200] + "...",
                    "score": float(node.score) if hasattr(node, 'score') else None,
                    "metadata": node.node.metadata
                })

        return QueryResponse(
            response=str(response),
            sources=sources,
            metadata={
                "top_k": request.top_k,
                "query_length": len(request.query)
            }
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Stream query response

    Returns server-sent events with streaming response
    """
    if query_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Query engine not initialized. Please ingest documents first."
        )

    async def generate():
        try:
            custom_query_engine = index.as_query_engine(
                similarity_top_k=request.top_k,
                streaming=True
            )

            response = custom_query_engine.query(request.query)

            for text in response.response_gen:
                yield f"data: {text}\n\n"
                await asyncio.sleep(0.01)  # Small delay for better streaming

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the assistant using RAG

    Maintains conversation context within a session
    """
    if chat_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Chat engine not initialized. Please ingest documents first."
        )

    try:
        logger.info(f"Processing chat message: {request.message}")

        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"

        # Get response
        response = chat_engine.chat(request.message)

        return ChatResponse(
            response=str(response),
            session_id=session_id,
            metadata={
                "message_length": len(request.message)
            }
        )

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response

    Returns server-sent events with streaming response
    """
    if chat_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Chat engine not initialized. Please ingest documents first."
        )

    async def generate():
        try:
            response = chat_engine.stream_chat(request.message)

            for text in response.response_gen:
                yield f"data: {text}\n\n"
                await asyncio.sleep(0.01)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error streaming chat: {e}")
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest documents into the index

    Can rebuild entire index or add new documents
    """
    try:
        if request.rebuild:
            logger.info("Rebuilding index...")
            background_tasks.add_task(rebuild_index)
            return {
                "status": "processing",
                "message": "Index rebuild started in background"
            }
        else:
            logger.info("Adding new documents...")
            background_tasks.add_task(add_documents, request.files)
            return {
                "status": "processing",
                "message": "Document ingestion started in background"
            }

    except Exception as e:
        logger.error(f"Error starting ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def rebuild_index():
    """Background task to rebuild entire index"""
    global index, query_engine, chat_engine

    try:
        logger.info("Starting index rebuild...")

        # Load documents
        documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
        logger.info(f"Loaded {len(documents)} documents")

        # Create new index
        index = VectorStoreIndex.from_documents(documents)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))

        # Update engines
        query_engine = index.as_query_engine(streaming=True)
        chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)

        logger.info("Index rebuild completed successfully")

    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise


def add_documents(file_paths: Optional[List[str]] = None):
    """Background task to add new documents"""
    global index, query_engine, chat_engine

    try:
        logger.info("Adding new documents...")

        if file_paths:
            # Load specific files
            documents = []
            for path in file_paths:
                file_path = DATA_DIR / path
                if file_path.exists():
                    docs = SimpleDirectoryReader(
                        input_files=[str(file_path)]
                    ).load_data()
                    documents.extend(docs)
        else:
            # Load all documents
            documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()

        logger.info(f"Loaded {len(documents)} documents")

        # Add to existing index or create new one
        if index is None:
            index = VectorStoreIndex.from_documents(documents)
        else:
            # Parse documents into nodes and insert
            from llama_index.core.node_parser import SimpleNodeParser
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(documents)
            index.insert_nodes(nodes)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))

        # Update engines
        query_engine = index.as_query_engine(streaming=True)
        chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)

        logger.info("Documents added successfully")

    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file

    Saves file to data directory for ingestion
    """
    try:
        file_path = DATA_DIR / file.filename

        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File uploaded: {file.filename}")

        return {
            "status": "success",
            "filename": file.filename,
            "size": len(content),
            "message": "File uploaded. Use /ingest endpoint to index it."
        }

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index")
async def clear_index():
    """
    Clear the index

    Removes all indexed data (files in data directory are preserved)
    """
    global index, query_engine, chat_engine

    try:
        # Clear storage directory
        import shutil
        if STORAGE_DIR.exists():
            shutil.rmtree(STORAGE_DIR)
            STORAGE_DIR.mkdir()

        # Reset global variables
        index = None
        query_engine = None
        chat_engine = None

        logger.info("Index cleared successfully")

        return {
            "status": "success",
            "message": "Index cleared. Data files preserved."
        }

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=(ENVIRONMENT == "development"),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
