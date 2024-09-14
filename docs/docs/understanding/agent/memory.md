# Memory

We've now made several additions and subtractions to our code. To make it clear what we're using, you can see [the current code for our agent](https://github.com/run-llama/python-agents-tutorial/blob/main/5_memory.py) in the repo. It's using OpenAI for the LLM and LlamaParse to enhance parsing.

We've also added 3 questions in a row. Let's see how the agent handles them:

```python
response = agent.chat(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)

print(response)

response = agent.chat(
    "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?"
)

print(response)

response = agent.chat(
    "How much was the total of those two allocations added together? Use a tool to answer any questions."
)

print(response)
```

This is demonstrating a powerful feature of agents in LlamaIndex: memory. Let's see what the output looks like:

```
Started parsing the file under job_id cac11eca-45e0-4ea9-968a-25f1ac9b8f99
Thought: The current language of the user is English. I need to use a tool to help me answer the question.
Action: canadian_budget_2023
Action Input: {'input': 'How much was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?'}
Observation: $20 billion was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget.
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: $20 billion was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget.
$20 billion was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget.
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: canadian_budget_2023
Action Input: {'input': 'How much was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget?'}
Observation: $13 billion was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget.
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: $13 billion was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget.
$13 billion was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget.
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: add
Action Input: {'a': 20, 'b': 13}
Observation: 33
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: The total of the allocations for the tax credit to promote investment in green technologies and the means-tested dental care program in the 2023 Canadian federal budget is $33 billion.
The total of the allocations for the tax credit to promote investment in green technologies and the means-tested dental care program in the 2023 Canadian federal budget is $33 billion.
```

The agent remembers that it already has the budget allocations from previous questions, and can answer a contextual question like "add those two allocations together" without needing to specify which allocations exactly. It even correctly uses the other addition tool to sum up the numbers.

Having demonstrated how memory helps, let's [add some more complex tools](./tools.md) to our agent.
