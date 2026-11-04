# Testing Setup

Integration tests need a local YDB:

```
docker run -d --name test-ydb -p 2136:2136 \
  -e YDB_USE_IN_MEMORY_PDISKS=true -h localhost \
  ydbplatform/local-ydb:trunk
```

Clean up: `docker stop test-ydb && docker rm test-ydb`
