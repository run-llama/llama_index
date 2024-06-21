from llama_index.llms.yi import Yi

# get api key from: https://platform.01.ai/
llm = Yi(model="yi-large", api_key="")

response = llm.chat("Hi, who are you?")
print(response)
