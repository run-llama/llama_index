# Testing Setup

## Installation

First, install the package:

```bash
pip install llama-index-vector-stores-paradedb
```

## Database Setup

You will need to start a postgres instance locally to run the tests for this integration. You can do this easily via docker cli:

```
docker run -d \
  --name paradedb \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=mark90 \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  paradedb/paradedb:latest
```

To clean up the created postgres DB, just run:

```
docker stop paradedb
docker rm paradedb
```
