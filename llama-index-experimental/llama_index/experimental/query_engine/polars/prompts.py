from llama_index.core.prompts import PromptTemplate, PromptType

############################################
# Polars
############################################

DEFAULT_POLARS_TMPL = (
    "You are working with a polars dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Important Polars syntax reminders:\n"
    "- Use pl.col('column_name') to reference columns\n"
    "- Use .select() for column selection: df.select([pl.col('col1'), pl.col('col2')])\n"
    "- Use .filter() for row filtering: df.filter(pl.col('col1') > 10)\n"
    "- Use .with_columns() to add/modify columns: df.with_columns([pl.col('col1').alias('new_col')])\n"
    "- For aggregations: df.group_by('col1').agg([pl.col('col2').sum()])\n"
    "- Import polars as pl if needed\n\n"
    "Expression:"
)

DEFAULT_POLARS_PROMPT = PromptTemplate(
    DEFAULT_POLARS_TMPL, prompt_type=PromptType.PANDAS  # Reusing PANDAS type as there's no POLARS type
)