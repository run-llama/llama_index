import openai

from llama_index.agent import OpenAIAgent
from llama_index.tools.finance_tools.base import FinanceAgentToolSpec

POLYGON_API_KEY = ""
FINNHUB_API_KEY = ""
ALPHA_VANTAGE_API_KEY = ""
NEWSAPI_API_KEY = ""
OPENAI_API_KEY = ""

GPT_MODEL_NAME = "gpt-4-0613"


def create_agent(
    polygon_api_key: str,
    finnhub_api_key: str,
    alpha_vantage_api_key: str,
    newsapi_api_key: str,
    openai_api_key: str,
) -> OpenAIAgent:
    openai.api_key = openai_api_key
    tool_spec = FinanceAgentToolSpec(
        polygon_api_key, finnhub_api_key, alpha_vantage_api_key, newsapi_api_key
    )
    llm = OpenAI(temperature=0, model=GPT_MODEL_NAME)
    return OpenAIAgent.from_tools(tool_spec.to_tool_list(), llm=llm, verbose=True)


if __name__ == "__main__":
    agent = create_agent(
        POLYGON_API_KEY,
        FINNHUB_API_KEY,
        ALPHA_VANTAGE_API_KEY,
        NEWSAPI_API_KEY,
        OPENAI_API_KEY,
    )
    print(agent.chat(sys.argv[1]))
