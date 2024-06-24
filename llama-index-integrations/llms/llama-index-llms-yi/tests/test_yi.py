from llama_index.llms.yi import Yi

# get api key from: https://platform.01.ai/
llm = Yi(model="yi-large", api_key="dd36902a1937481d9984227e465c1c88")

response = llm.chat("Hi, who are you?")
print(response)
