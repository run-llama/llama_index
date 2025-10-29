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
    "CRITICAL Polars syntax rules - follow these exactly:\n"
    "- ALWAYS start with 'df.' - NEVER start with pl.col() alone\n"
    "- pl.col() creates expressions that must be used INSIDE DataFrame methods\n"
    "- pl.col() expressions do NOT have .to_frame() method - this does not exist!\n"
    "- For selecting: df.select([pl.col('col1'), pl.col('col2')])\n"
    "- For filtering: df.filter(pl.col('col1') > 10)\n"
    "- For sorting: df.sort('column', descending=True) or df.sort(pl.col('column'), descending=True)\n"
    "- For grouping: df.group_by('col1').agg([pl.col('col2').sum()]) NOT .groupby()\n"
    "- For limiting: df.limit(5) or df.head(5)\n"
    "- For aliasing: pl.col('old_name').alias('new_name') (only inside select/agg)\n"
    "- WRONG: pl.col('company').to_frame() - this does NOT exist\n"
    "- CORRECT: df.select([pl.col('company'), pl.col('revenue')])\n"
    "- Always use complete, executable expressions that return a dataframe\n"
    "- Do NOT assign to variables - return the final dataframe expression\n"
    "- Keep expressions simple - avoid complex multi-line chaining\n"
    "- Example: df.select([pl.col('company'), pl.col('revenue')]).sort('revenue', descending=True).limit(5)\n\n"
    "Generate a single Polars expression that answers the query. Return ONLY the code, no explanations:\n"
)

DEFAULT_POLARS_PROMPT = PromptTemplate(
    DEFAULT_POLARS_TMPL,
    prompt_type=PromptType.PANDAS,  # Reusing PANDAS type as there's no POLARS type
)
