# Quick Start Guide

Get your LlamaIndex application running in under 5 minutes!

## Prerequisites

Choose your deployment method:

**Option 1 - Local (Python):**
- Python 3.9 or higher
- pip (Python package manager)
- OpenAI API key

**Option 2 - Docker:**
- Docker installed
- OpenAI API key

**Option 3 - Docker Compose (Recommended):**
- Docker and Docker Compose installed
- OpenAI API key

---

## 🚀 Fastest Way: Automated Setup

```bash
# Run the interactive setup script
./start.sh
```

The script will:
- Check prerequisites
- Set up environment variables
- Install dependencies
- Create sample data
- Start the server

Then follow the prompts!

---

## Option 1: Local Python Deployment

### Step 1: Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# Change this line:
# OPENAI_API_KEY=your-key-here
# To:
# OPENAI_API_KEY=sk-your-actual-key
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Add Your Documents

```bash
# Add some text files to the data directory
echo "LlamaIndex is a data framework for LLM applications." > data/sample.txt
# Or copy your own documents
cp /path/to/your/documents/*.txt data/
```

### Step 4: Start Server

```bash
python api_server.py
```

### Step 5: Test It!

Visit http://localhost:8000/docs in your browser

Or use curl:
```bash
# Query your documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LlamaIndex?"}'
```

---

## Option 2: Docker Deployment

### Step 1: Setup Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 2: Add Documents

```bash
# Create sample document
echo "LlamaIndex is amazing!" > data/sample.txt
```

### Step 3: Build and Run

```bash
# Build image
docker build -t llamaindex-app .

# Run container
docker run -d \
  --name llamaindex-app \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/storage:/app/storage \
  llamaindex-app
```

### Step 4: Check Status

```bash
# View logs
docker logs -f llamaindex-app

# Check health
curl http://localhost:8000/health
```

---

## Option 3: Docker Compose (Full Stack) ⭐ Recommended

### Step 1: Setup

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 2: Add Documents

```bash
echo "LlamaIndex deployment guide" > data/sample.txt
```

### Step 3: Start Everything

```bash
docker-compose up -d
```

This starts:
- FastAPI application (port 8000)
- Chroma vector database (port 8001)
- PostgreSQL (port 5432)
- Redis cache (port 6379)

### Step 4: Check Status

```bash
# View all services
docker-compose ps

# View logs
docker-compose logs -f app

# Check health
curl http://localhost:8000/health
```

---

## Your First Query

### Using the Web Interface

1. Open http://localhost:8000/docs
2. Find the `/query` endpoint
3. Click "Try it out"
4. Enter your query
5. Click "Execute"

### Using cURL

```bash
# Simple query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are these documents about?",
    "top_k": 5
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is LlamaIndex?",
        "top_k": 5
    }
)

print(response.json())
```

---

## Common Tasks

### Upload a Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```

### Index Documents

```bash
# Index new documents
curl -X POST http://localhost:8000/ingest

# Rebuild entire index
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"rebuild": true}'
```

### Chat Interface

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the documents",
    "session_id": "user123"
  }'
```

### Streaming Responses

```bash
# Stream query response
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain in detail"}' \
  --no-buffer
```

---

## Troubleshooting

### "OPENAI_API_KEY not set" Error

```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Make sure it starts with sk-
# OPENAI_API_KEY=sk-your-actual-key
```

### Port 8000 Already in Use

```bash
# Find what's using port 8000
lsof -i :8000  # On macOS/Linux
netstat -ano | findstr :8000  # On Windows

# Kill the process or use a different port
# In .env, change:
PORT=8001
```

### Index Not Loading

```bash
# Clear and rebuild
rm -rf storage/*
curl -X POST http://localhost:8000/ingest -d '{"rebuild": true}'
```

### Docker Issues

```bash
# Check Docker is running
docker ps

# Restart services
docker-compose restart

# View detailed logs
docker-compose logs --tail=100 app

# Clean restart
docker-compose down -v
docker-compose up -d
```

### "No documents found" Error

```bash
# Check data directory
ls -la data/

# Add a test document
echo "Test document" > data/test.txt

# Trigger indexing
curl -X POST http://localhost:8000/ingest
```

---

## Makefile Commands

If you have `make` installed, you can use these convenient commands:

```bash
make help          # Show all commands
make setup         # Initial setup
make install       # Install dependencies
make run           # Run server
make docker-up     # Start Docker services
make docker-logs   # View logs
make health        # Check health
make clean         # Clean up
```

See `Makefile` for all available commands.

---

## What's Next?

### For Development

1. Read the full [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Customize `api_server.py` for your needs
3. Add more vector stores (Pinecone, Qdrant, etc.)
4. Implement authentication
5. Add rate limiting

### For Production

1. Set `ENVIRONMENT=production` in `.env`
2. Use a managed vector database (Pinecone, Weaviate, etc.)
3. Set up SSL/TLS with nginx
4. Implement authentication and authorization
5. Add monitoring (Prometheus, Grafana)
6. Set up logging aggregation
7. Configure auto-scaling
8. Set up CI/CD pipeline

### Deployment Guides

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for:
- AWS deployment (EC2, ECS, Lambda)
- GCP deployment (Cloud Run, Compute Engine)
- Azure deployment
- Kubernetes deployment
- Best practices and optimization

---

## Need Help?

- Check the full [README.md](README.md)
- Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Visit [LlamaIndex Docs](https://docs.llamaindex.ai/)
- Join [LlamaIndex Discord](https://discord.gg/dGcwcsnxhU)

---

## Summary

**You just:**
- ✅ Set up a production-ready LlamaIndex API
- ✅ Indexed your documents
- ✅ Created a queryable knowledge base
- ✅ Deployed with Docker (optional)

**In less than 5 minutes!** 🎉

Now go build something amazing! 🚀
