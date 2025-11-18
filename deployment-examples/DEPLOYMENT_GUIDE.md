# LlamaIndex Deployment Guide

Complete guide for deploying LlamaIndex applications locally and on production servers.

## Table of Contents

1. [Understanding LlamaIndex](#understanding-llamaindex)
2. [Local Deployment](#local-deployment)
3. [Production Deployment](#production-deployment)
4. [Deployment Platforms](#deployment-platforms)
5. [Best Practices](#best-practices)

---

## Understanding LlamaIndex

**Important:** LlamaIndex is a **library/framework**, not a standalone application. You need to:
- Create your own application (FastAPI, Flask, Streamlit, etc.)
- Import LlamaIndex components into your app
- Build, containerize, and deploy YOUR application

---

## Local Deployment

### Option 1: Simple Python Application (Quickstart)

#### Prerequisites
```bash
# Python 3.9+ required
python --version

# Install uv (recommended) or pip
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR use pip
```

#### Step 1: Create Project Structure
```bash
mkdir my-llamaindex-app
cd my-llamaindex-app
mkdir data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
# Using uv (faster)
uv pip install llama-index llama-index-llms-openai llama-index-embeddings-openai

# OR using pip
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
```

#### Step 3: Create Simple App
Create `simple_app.py`:
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

# Configure LLM and embeddings
Settings.llm = OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
print(response)
```

#### Step 4: Run Locally
```bash
export OPENAI_API_KEY="your-key-here"
python simple_app.py
```

---

### Option 2: FastAPI Server (Recommended for Production)

See the example FastAPI application in this directory:
- `api_server.py` - Main FastAPI application
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

#### Run FastAPI Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

Access at: http://localhost:8000
API docs: http://localhost:8000/docs

---

### Option 3: Streamlit Web App

LlamaIndex provides a Streamlit chatbot pack:

```bash
# Install Streamlit pack
pip install llama-index-packs-streamlit-chatbot

# Create app
llamaindex-cli download-llamapack StreamlitChatbot --download-dir ./streamlit_app

cd streamlit_app
streamlit run base.py
```

---

## Production Deployment

### Docker Deployment (Recommended)

#### Prerequisites
- Docker installed
- Docker Compose installed (for local testing)

#### Step 1: Build Docker Image

Using the provided `Dockerfile`:

```bash
# Build image
docker build -t my-llamaindex-app:latest .

# Test locally
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  my-llamaindex-app:latest
```

#### Step 2: Docker Compose (Full Stack)

The provided `docker-compose.yml` includes:
- FastAPI application
- Chroma vector database
- Optional PostgreSQL

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

#### Step 3: Production Configuration

**Environment Variables:**
```bash
# Required
OPENAI_API_KEY=sk-...
ENVIRONMENT=production

# Optional - Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=chroma
CHROMA_PORT=8000

# Optional - Database
DATABASE_URL=postgresql://user:pass@db:5432/llamaindex

# Optional - Performance
WORKERS=4
LOG_LEVEL=info
```

**Docker Compose Production:**
```yaml
version: '3.8'

services:
  app:
    build: .
    restart: always
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
      - WORKERS=4
    volumes:
      - ./data:/app/data
      - app-cache:/app/.cache
    depends_on:
      - chroma
      - db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  chroma:
    image: chromadb/chroma:latest
    restart: always
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  db:
    image: postgres:15-alpine
    restart: always
    environment:
      - POSTGRES_USER=llamaindex
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=llamaindex
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  chroma-data:
  postgres-data:
  app-cache:
```

---

## Deployment Platforms

### 1. AWS Deployment

#### AWS EC2
```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone your repo
git clone your-repo.git
cd your-repo

# Set environment variables
nano .env  # Add your secrets

# Run with Docker Compose
docker-compose up -d

# Setup nginx reverse proxy (optional)
sudo apt install nginx
# Configure nginx to proxy to port 8000
```

#### AWS ECS (Elastic Container Service)
```bash
# Build and push to ECR
aws ecr create-repository --repository-name llamaindex-app
docker tag my-llamaindex-app:latest YOUR_ECR_URL/llamaindex-app:latest
docker push YOUR_ECR_URL/llamaindex-app:latest

# Create ECS task definition and service
# Use provided ecs-task-definition.json
```

#### AWS Lambda + API Gateway
- Best for low-traffic applications
- Use Docker container support in Lambda
- Max timeout: 15 minutes
- Consider cold starts

### 2. Google Cloud Platform

#### GCP Cloud Run (Easiest)
```bash
# Install gcloud CLI
gcloud init

# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/llamaindex-app
gcloud run deploy llamaindex-app \
  --image gcr.io/PROJECT_ID/llamaindex-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key
```

#### GCP Compute Engine (VM)
- Similar to AWS EC2
- Use startup scripts for automation

### 3. Azure Deployment

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name llamaindex-app \
  --image myregistry.azurecr.io/llamaindex-app:latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=your-key
```

#### Azure App Service
- Deploy directly from Docker Hub or ACR
- Built-in scaling and monitoring

### 4. Kubernetes (Any Cloud)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaindex-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llamaindex-app
  template:
    metadata:
      labels:
        app: llamaindex-app
    spec:
      containers:
      - name: app
        image: your-registry/llamaindex-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: llamaindex-service
spec:
  selector:
    app: llamaindex-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Apply:
```bash
kubectl apply -f deployment.yaml
```

### 5. DigitalOcean App Platform

```yaml
# .do/app.yaml
name: llamaindex-app
services:
- name: web
  github:
    repo: your-username/your-repo
    branch: main
  dockerfile_path: Dockerfile
  http_port: 8000
  instance_count: 2
  instance_size_slug: basic-s
  envs:
  - key: OPENAI_API_KEY
    scope: RUN_TIME
    type: SECRET
    value: "${OPENAI_API_KEY}"
  - key: ENVIRONMENT
    value: "production"
```

### 6. Heroku

```bash
# Install Heroku CLI
heroku login

# Create app
heroku create my-llamaindex-app

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key

# Deploy
git push heroku main
```

### 7. Railway.app / Render.com

Both support:
- Direct GitHub integration
- Automatic deployments
- Docker or buildpack
- Easy environment variable management

---

## Best Practices

### 1. Security

**Environment Variables:**
```bash
# NEVER commit secrets to git
# Use .env files (gitignored)
# Use cloud secret managers in production
```

**API Keys:**
- AWS Secrets Manager
- GCP Secret Manager
- Azure Key Vault
- HashiCorp Vault

**Authentication:**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials
```

### 2. Performance

**Caching:**
```python
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage

# Save index
index.storage_context.persist(persist_dir="./storage")

# Load cached index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

**Vector Databases:**
- Use managed vector DBs (Pinecone, Weaviate, Qdrant Cloud)
- Or self-host: Chroma, Milvus, Postgres+pgvector

**Connection Pooling:**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### 3. Monitoring

**Logging:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Metrics:**
- Prometheus + Grafana
- DataDog
- New Relic
- CloudWatch (AWS)

**Health Checks:**
```python
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/ready")
async def ready():
    # Check database, vector store connections
    return {"status": "ready"}
```

### 4. Scaling

**Horizontal Scaling:**
- Use load balancers
- Stateless application design
- Shared vector database
- Session storage in Redis

**Vertical Scaling:**
- Monitor CPU/Memory usage
- Increase container resources
- Use GPU instances for embeddings

**Async Processing:**
```python
from fastapi import BackgroundTasks

@app.post("/ingest")
async def ingest_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_documents)
    return {"status": "processing"}
```

### 5. Cost Optimization

**LLM Costs:**
- Use cheaper models for simple tasks
- Implement caching (responses, embeddings)
- Batch processing
- Set max tokens limits

**Infrastructure:**
- Use autoscaling
- Spot instances for dev/test
- Reserved instances for production
- Monitor usage with alerts

### 6. Data Management

**Document Ingestion:**
```python
# Batch processing
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        embed_model,
    ]
)

nodes = pipeline.run(documents=documents)
```

**Incremental Updates:**
```python
# Add new documents without rebuilding entire index
index.insert_nodes(new_nodes)
index.storage_context.persist()
```

### 7. Testing

**Unit Tests:**
```python
import pytest
from api_server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_query_endpoint():
    response = client.post(
        "/query",
        json={"query": "test question"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

**Load Testing:**
```bash
# Using locust
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory:**
- Reduce chunk size
- Use streaming responses
- Increase container memory
- Implement pagination

**2. Slow Response Times:**
- Enable caching
- Use faster embedding models
- Optimize vector search parameters
- Add response streaming

**3. API Rate Limits:**
- Implement retry logic with backoff
- Use multiple API keys (rotation)
- Cache frequently used responses
- Monitor usage

**4. Vector Store Connection:**
- Check network connectivity
- Verify credentials
- Check firewall rules
- Use connection pooling

---

## Next Steps

1. **Start Small:** Deploy the simple FastAPI example locally
2. **Add Features:** Integrate your specific data sources
3. **Test Thoroughly:** Load test before production
4. **Monitor:** Set up logging and metrics
5. **Scale:** Add caching, CDN, load balancing as needed
6. **Iterate:** Continuously optimize based on usage patterns

---

## Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---

## Support

For LlamaIndex issues:
- GitHub: https://github.com/run-llama/llama_index
- Discord: https://discord.gg/dGcwcsnxhU
- Documentation: https://docs.llamaindex.ai/
