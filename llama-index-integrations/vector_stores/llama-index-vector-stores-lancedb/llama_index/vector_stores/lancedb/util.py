from llama_index.core.vector_stores.types import (
    FilterOperator,
)

sql_operator_mapper = {
    FilterOperator.EQ: " = ",
    FilterOperator.GT: " > ",
    FilterOperator.GTE: " >= ",
    FilterOperator.LTE: " <= ",
    FilterOperator.TEXT_MATCH: " LIKE ",
    FilterOperator.NE: " NOT LIKE ",
    FilterOperator.IN: " IN ",
}
