# LlamaIndex Deployment Examples

Production-ready examples for deploying LlamaIndex applications locally and on servers.

## Quick Start

### Local Deployment (3 minutes)

```bash
# 1. Copy environment file and add your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add some documents to the data folder
echo "This is a test document about deployment." > data/test.txt

# 4. Run the server
python api_server.py
```

Visit http://localhost:8000/docs for interactive API documentation.

### Docker Deployment (5 minutes)

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Add documents to data folder
echo "Test document" > data/test.txt

# 3. Start all services
docker-compose up -d

# 4. Check logs
docker-compose logs -f app

# 5. Test the API
curl http://localhost:8000/health
```

## Project Structure

```
deployment-examples/
├── api_server.py           # FastAPI application with LlamaIndex
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Multi-container setup
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
├── nginx.conf            # Nginx reverse proxy config
├── DEPLOYMENT_GUIDE.md   # Comprehensive deployment guide
├── data/                 # Document storage (your files)
└── storage/              # Index persistence (auto-generated)
```

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /docs` - Interactive API documentation

### Query Endpoints

- `POST /query` - Query documents (RAG)
  ```json
  {
    "query": "What is LlamaIndex?",
    "top_k": 5,
    "stream": false
  }
  ```

- `POST /query/stream` - Streaming query response

### Chat Endpoints

- `POST /chat` - Chat with context
  ```json
  {
    "message": "Tell me about deployment",
    "session_id": "user123",
    "stream": false
  }
  ```

- `POST /chat/stream` - Streaming chat response

### Document Management

- `POST /upload` - Upload a document file
- `POST /ingest` - Index uploaded documents
- `DELETE /index` - Clear the index

## Usage Examples

### Query Documents

```bash
# Basic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?"}'

# Query with custom top_k
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain deployment", "top_k": 3}'
```

### Upload and Index Documents

```bash
# Upload a file
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"

# Trigger indexing
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"rebuild": false}'
```

### Chat

```bash
# Chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What can you tell me about the documents?",
    "session_id": "user-123"
  }'
```

### Streaming Response

```bash
# Stream query response
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain in detail"}' \
  --no-buffer
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Important:**
- `ENVIRONMENT` - development, staging, or production
- `LLM_MODEL` - Which GPT model to use
- `CHUNK_SIZE` - Document chunk size for indexing
- `VECTOR_STORE_TYPE` - Vector database (chroma, pinecone, etc.)

### Vector Stores

The example supports multiple vector stores:

**In-Memory (Default):**
- No additional setup required
- Data lost on restart
- Good for testing

**Chroma (Included in docker-compose):**
```bash
# Already configured in docker-compose.yml
docker-compose up -d chroma
```

**Pinecone:**
```bash
# Add to .env
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=llamaindex

# Update requirements.txt
# llama-index-vector-stores-pinecone>=0.4.0
```

**Other options:** Qdrant, Milvus, PostgreSQL+pgvector

## Production Deployment

### Option 1: Docker on VM (AWS EC2, GCP Compute, etc.)

```bash
# SSH into server
ssh user@your-server

# Clone repo
git clone your-repo.git
cd your-repo/deployment-examples

# Setup environment
cp .env.example .env
nano .env  # Add your secrets

# Run with docker-compose
docker-compose up -d

# Setup nginx reverse proxy (optional)
sudo apt install nginx
sudo cp nginx.conf /etc/nginx/sites-available/llamaindex
sudo ln -s /etc/nginx/sites-available/llamaindex /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### Option 2: Cloud Run (GCP)

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/llamaindex-app
gcloud run deploy llamaindex-app \
  --image gcr.io/PROJECT_ID/llamaindex-app \
  --platform managed \
  --set-env-vars OPENAI_API_KEY=your-key
```

### Option 3: AWS ECS

```bash
# Push to ECR
aws ecr create-repository --repository-name llamaindex-app
docker tag llamaindex-app:latest YOUR_ECR_URL/llamaindex-app:latest
docker push YOUR_ECR_URL/llamaindex-app:latest

# Deploy using ECS console or CLI
```

### Option 4: Kubernetes

```bash
# Create secret
kubectl create secret generic api-secrets \
  --from-literal=openai-key=your-key

# Deploy
kubectl apply -f k8s-deployment.yaml
```

## Monitoring

### Docker Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app

# Last 100 lines
docker-compose logs --tail=100 app
```

### Health Checks

```bash
# Health endpoint
curl http://localhost:8000/health

# Readiness check (fails if index not loaded)
curl http://localhost:8000/ready

# Container health status
docker-compose ps
```

### Metrics (Optional)

Add Prometheus and Grafana for monitoring:

```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
```

### Load Balancing

Add nginx or cloud load balancer:

```nginx
upstream llamaindex_backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    location / {
        proxy_pass http://llamaindex_backend;
    }
}
```

## Troubleshooting

### Application won't start

```bash
# Check logs
docker-compose logs app

# Common issues:
# 1. Missing OPENAI_API_KEY in .env
# 2. Port 8000 already in use
# 3. Insufficient memory
```

### Index not loading

```bash
# Clear storage and rebuild
rm -rf storage/*
curl -X POST http://localhost:8000/ingest -d '{"rebuild": true}'
```

### Out of memory

```bash
# Reduce chunk size in .env
CHUNK_SIZE=512

# Increase Docker memory limit
# In docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Slow queries

```bash
# Use faster embedding model
EMBEDDING_MODEL=text-embedding-3-small

# Reduce top_k
SIMILARITY_TOP_K=3

# Enable caching (add Redis)
docker-compose up -d redis
```

## Security Best Practices

1. **Never commit secrets**
   - Use `.env` files (gitignored)
   - Use secret managers in production

2. **Add authentication**
   - Implement API key validation
   - Use OAuth2/JWT for user auth

3. **Use HTTPS**
   - Set up SSL certificates
   - Use nginx with Let's Encrypt

4. **Rate limiting**
   - Implement per-user rate limits
   - Use Redis for rate limit tracking

5. **Input validation**
   - Validate all user inputs
   - Sanitize file uploads

## Next Steps

1. Read the [comprehensive deployment guide](DEPLOYMENT_GUIDE.md)
2. Customize `api_server.py` for your use case
3. Add authentication and rate limiting
4. Set up monitoring and alerts
5. Deploy to your chosen platform

## Support

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

## License

This example is provided as-is for educational and development purposes.
