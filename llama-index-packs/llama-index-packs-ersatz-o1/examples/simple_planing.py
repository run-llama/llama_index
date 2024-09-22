import asyncio

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from llama_index.packs.ersatz_o1 import ErsatzO1QueryEngine

llm = OpenAI(model="gpt-4-turbo")

task_context = """
Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables
to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of
chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed.  
"""

query_engine = ErsatzO1QueryEngine(
    context=task_context,
    llm=llm,
    reasoning_paths=5,
    verbose=True,
)

if __name__ == "__main__":
    res = query_engine.query("How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi’s flock is 20 chickens?")
    print(res)

