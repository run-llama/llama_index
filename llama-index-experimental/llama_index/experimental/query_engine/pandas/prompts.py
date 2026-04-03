from llama_index.core.prompts import PromptTemplate, PromptType

############################################
# Pandas
############################################

DEFAULT_PANDAS_TMPL = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

DEFAULT_PANDAS_PROMPT = PromptTemplate(
    DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
)
