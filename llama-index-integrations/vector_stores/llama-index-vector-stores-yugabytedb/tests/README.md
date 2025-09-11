# Testing Setup

You will need to start a Yugabytedb instance locally to run the tests for this integration. You can do this easily via docker cli:

```
./bin/yugabyted start
```

For more information about starting a Yugabytedb cluster, see [here](https://docs.yugabyte.com/preview/tutorials/quick-start/macos/).

Run the tests:

```
 uv run -- pytest
```