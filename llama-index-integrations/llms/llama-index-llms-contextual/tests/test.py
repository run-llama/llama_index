from llama_index.llms.contextual import Contextual

# Set up the Contextual class with the required model and API key
llm = Contextual(model="contextual-clm", api_key="key-47EdcA3Tim9bif-W96XAsA0nFPf9ZPvkwtE4cL7vP5s7NtAHw")

# Call the complete method with a query
response = llm.complete("Explain the importance of low latency LLMs")

print(response)