"""
Prompts for implementing Chain of Abstraction.

While official prompts are not given (and the paper finetunes models for the task),
we can take inspiration and use few-shot prompting to generate a prompt for implementing
chain of abstraction in an LLM agent.
"""

REASONING_PROMPT_TEMPALTE = """Generate an abstract plan of reasoning using placeholders for the specific values and function calls needed.
The placeholders should be labeled y1, y2, etc.
Function calls should be represented as inline strings like [FUNC {{function_name}}({{input1}}, {{input2}}, ...) = {{output_placeholder}}].
Assume someone will read the plan after the functions have been executed in order to make a final response.
Not every question will require function calls to answer.
If you do invoke a function, only use the available functions, do not make up functions.

Example:
-----------
Available functions:
```python
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    ...

def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers together.\"\"\"
    ...
```

Question:
Sally has 3 apples and buys 2 more. Then magically, a wizard casts a spell that multiplies the number of apples by 3. How many apples does Sally have now?

Abstract plan of reasoning:
After buying the apples, Sally has [FUNC add(3, 2) = y1] apples. Then, the wizard casts a spell to multiply the number of apples by 3, resulting in [FUNC multiply("y1", 3) = y2] apples.

Your Turn:
-----------
Available functions:
```python
{functions}
```

Question:
{question}

Abstract plan of reasoning:
"""

REFINE_REASONING_PROMPT_TEMPALTE = """Generate a response to a question by using a previous abstract plan of reasoning. Use the previous reasoning as context to write a response to the question.

Example:
-----------
Question:
Sally has 3 apples and buys 2 more. Then magically, a wizard casts a spell that multiplies the number of apples by 3. How many apples does Sally have now?

Previous reasoning:
After buying the apples, Sally has [FUNC add(3, 2) = 5] apples. Then, the wizard casts a spell to multiply the number of apples by 3, resulting in [FUNC multiply(5, 3) = 15] apples.

Response:
After the wizard casts the spell, Sally has 15 apples.

Your Turn:
-----------
Question:
{question}

Previous reasoning:
{prev_reasoning}

Response:
"""
