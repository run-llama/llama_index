from typing import Any


def _import_databricks() -> Any:
    """
    Try to import databricks.vector_search.client.VectorSearchIndex. If databricks module it's not already installed, instruct user how to install.
    """

    try:
        from databricks.vector_search.client import VectorSearchIndex
    except ImportError:
        raise ImportError(
            "`databricks-vectorsearch` package not found: "
            "please run `pip install databricks-vectorsearch`"
        )
