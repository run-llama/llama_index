"""Test pandas index."""

from typing import Any, Dict, cast

import pandas as pd
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.query_engine.pandas_query_engine import PandasQueryEngine


def test_pandas_query_engine(mock_service_context: ServiceContext) -> None:
    """Test pandas query engine."""
    # Test on some sample data
    df = pd.DataFrame(
        {
            "city": ["Toronto", "Tokyo", "Berlin"],
            "population": [2930000, 13960000, 3645000],
            "description": [
                """Toronto, Canada's largest city, is a vibrant and diverse metropolis situated in the province of Ontario.
Known for its iconic skyline featuring the CN Tower, Toronto is a cultural melting pot with a rich blend of communities, languages, and cuisines.
It boasts a thriving arts scene, world-class museums, and a strong economic hub.
Visitors can explore historic neighborhoods, such as Kensington Market and Distillery District, or enjoy beautiful natural surroundings on Toronto Islands.
With its welcoming atmosphere, top-notch education, and multicultural charm, Toronto is a global destination for both tourists and professionals alike.""",
                "A city",
                "Another City",
            ],
        }
    )
    # the mock prompt just takes the all items in the given column
    query_engine = PandasQueryEngine(
        df, service_context=mock_service_context, verbose=True
    )
    response = query_engine.query(QueryBundle("population"))
    import sys

    if sys.version_info < (3, 9):
        assert str(response) == 'df["population"]'
    else:
        assert str(response) == str(df["population"])
    metadata = cast(Dict[str, Any], response.metadata)
    assert metadata["pandas_instruction_str"] == ('df["population"]')

    query_engine = PandasQueryEngine(
        df,
        service_context=mock_service_context,
        verbose=True,
        output_kwargs={"max_colwidth": 90},
    )
    response = query_engine.query(QueryBundle("description"))
    if sys.version_info < (3, 9):
        assert str(response) == 'df["description"]'
    else:
        pd.set_option("display.max_colwidth", 90)
        correst_rsp_str = str(df["description"])
        pd.reset_option("display.max_colwidth")
        assert str(response) == correst_rsp_str
