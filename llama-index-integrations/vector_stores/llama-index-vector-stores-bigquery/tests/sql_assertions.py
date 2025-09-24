import sqlparse


def assert_equivalent_sql_statements(actual_query: str, expected_query: str):
    def standardize_format(q: str) -> str:
        """Standardize SQL formatting for more reliable string comparison."""
        formatted = sqlparse.format(
            q,
            strip_comments=True,
            reindent=True,
            indent_tabs=False,
        )
        return " ".join(formatted.lower().split())

    formatted_query = standardize_format(actual_query)
    formatted_expected = standardize_format(expected_query)

    assert formatted_query == formatted_expected, (
        f"\n[Actual Query]:\n{formatted_query}\n\n"
        f"[Expected Query]:\n{formatted_expected}\n"
    )
