"""
This example demonstrates how to set up and test chatting with a finance agent using the FinanceChatPack.
It involves collecting necessary API keys and initializing the FinanceChatPack with these keys and a PostgreSQL database URI.
"""

import getpass
from llama_index.packs.finchat import FinanceChatPack

# Prompting the user to enter all necessary API keys for finance data access and OpenAI
openai_api_key = getpass.getpass("Enter your OpenAI API key: ")
polygon_api_key = getpass.getpass("Enter your Polygon API key: ")
finnhub_api_key = getpass.getpass("Enter your Finnhub API key: ")
alpha_vantage_api_key = getpass.getpass("Enter your Alpha Vantage API key: ")
newsapi_api_key = getpass.getpass("Enter your NewsAPI API key: ")

# PostgreSQL database URI for storing and accessing financial data
postgres_db_uri = "postgresql://postgres.xhlcobfkbhtwmckmszqp:fingptpassword#123@aws-0-us-east-1.pooler.supabase.com:5432/postgres"

# Initializing the FinanceChatPack with the collected API keys and database URI
finance_chat_pack = FinanceChatPack(
    polygon_api_key=polygon_api_key,
    finnhub_api_key=finnhub_api_key,
    alpha_vantage_api_key=alpha_vantage_api_key,
    newsapi_api_key=newsapi_api_key,
    openai_api_key=openai_api_key,
    postgres_db_uri=postgres_db_uri,
)

# Notifying the user that the FinanceChatPack has been initialized and is ready for testing
print(
    "FinanceChatPack initialized successfully. Ready for testing chat interactions with the finance agent."
)


user_query = "Find similar companies to Rivian?"
response = finance_chat_pack.run(user_query)
print("Finance agent response:", response)
