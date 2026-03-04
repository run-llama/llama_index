# Testing Setup

You will need to start a postgres instance locally to run the tests for this integration. You can do this easily via docker cli:

```
docker run --name test-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=mark90 \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg17
```

To clean up the created postgres DB, just run:

```
docker stop test-postgres
docker rm test-postgres
```
