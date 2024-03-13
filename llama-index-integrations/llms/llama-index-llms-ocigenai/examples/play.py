from llama_index.llms.ocigenai import OCIGenAI
import oci
from llama_index.core.llms import ChatMessage


llm = OCIGenAI(
        model="cohere.command", # "meta.llama-2-70b-chat" or "cohere.command"
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.tenancy.oc1..aaaaaaaasz6cicsgfbqh6tj3xahi4ozoescfz36bjm3kucc7lotk2oqep47q",
        #provider='cohere',
        #additional_kwargs={"temperature": 0, "max_tokens": 512, "top_p": 0.7, "frequency_penalty": 1.0}
        )

#complete
resp= llm.complete("Paul Graham is ")
print(resp)
exit()

# stream complete
# resp= llm.stream_complete("Paul Graham is ")
# for r in resp:
#     print(r.delta, end="")

# chat
messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="Tell me a story"),
]

#resp = llm.chat(messages)
#print(resp)

# stream chat
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")

# asynchronous complete
async def complete(prompt):
    resp = await llm.acomplete(prompt, formatted=True)
    print(resp)
    return resp

import asyncio
asyncio.run(complete("Paul Graham is "))