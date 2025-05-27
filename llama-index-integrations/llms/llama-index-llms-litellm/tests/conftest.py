# there's a rather large issue with the pants build, it's only running tests
# with sources that are imported, which causes pytest markers to not be registered
# so we need to import pytest_asyncio manually here to ensure that the markers
# are registered
import pytest_asyncio  # noqa: F401
