# Usage and Contribution Information for Bloomberg Users

This file contains internal usage instructions, and **should not** be published
as part of open-source contributions in public GitHub.

## Syncing the `uv.lock` File

When syncing the virtual environment with `uv`, we want to be careful to use our
internal PyPI mirror, but to _commit_ a `uv.lock` file that is compatible with
public PyPI.

To update the public `uv.lock` file, first **disconnect from BBVPN**, then do
the following:

```shell
# unset any env vars that would cause uv to use internal mirrors
unset UV_INSTALLER_GHE_BASE_URL UV_PYTHON_INSTALL_MIRROR UV_INDEX

# only if ~/.config/uv/uv.toml is present
mv ~/.config/uv/uv.toml ~/.config/uv/uv.toml.bak

# run uv sync to update uv.lock
uv sync --native-tls

# only if moved above
mv ~/.config/uv/uv.toml.bak ~/.config/uv/uv.toml
```

## Docker for Tests

The [docker-compose file](./tests/docker-compose.yml) is used to run a Solr
cloud for integration tests. Because Docker Hub images cannot be used directly
while on BBVPN, you need to do a slight workaround to make the image play nice.
For the two images needed (`solr:9-slim` and `zookeeper:3.9`), you can pull them
from the internal mirror and tag them so that `docker compose` can find them
without editing the compose file.

```shell
docker pull artprod.dev.bloomberg.com/ds/ext/registry-1.docker.io/library/solr:9-slim
docker image tag artprod.dev.bloomberg.com/ds/ext/registry-1.docker.io/library/solr:9-slim solr:9-slim

docker pull artprod.dev.bloomberg.com/ds/ext/registry-1.docker.io/library/zookeeper:3.9
docker image tag artprod.dev.bloomberg.com/ds/ext/registry-1.docker.io/library/zookeeper:3.9 zookeeper:3.9
```
