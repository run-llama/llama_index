from llama_index.llms.oci_genai import OCIGenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput


# Define a simple multiplication function
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


# Create a wrapper tool for the callable
class CallableTool(BaseTool):
    def __init__(self, func):
        self.func = func

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.func.__name__,
            description=self.func.__doc__,
        )

    def __call__(self, input: dict) -> ToolOutput:
        # Expecting input as a dictionary with keys 'a' and 'b'
        a = input.get('a')
        b = input.get('b')
        result = self.func(a, b)
        return ToolOutput(content=str(result), tool_name=self.metadata.name, raw_input=input, raw_output=result)


# Define the chat messages
messages = [
    ChatMessage(role="user", content="Can you multiply 5 and 3?")
]

# Create an instance of the OCIGenAI model
llm = OCIGenAI(
    model="cohere.command-r-plus",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaumuuscymm6yb3wsbaicfx3mjhesghplvrvamvbypyehh5pgaasna",  # Replace with your compartment ID
    auth_type="SECURITY_TOKEN",
    auth_profile="BoatOc1",
    additional_kwargs={"top_p": 0.7, "top_k": 5}
)

# Create an instance of the CallableTool with the multiply function
tool = CallableTool(multiply)

# Use chat_with_tools method to invoke the tool within the chat response
response = llm.chat_with_tools(
    tools=[tool],
    user_msg="Can you multiply 5 and 3?",
    chat_history=messages,
)

# Print the response
print("Response message:", response)

# Directly calling the tool for verification
input_data = {'a': 5, 'b': 3}
tool_output = tool(input_data)
print("Tool output (multiply):", tool_output.content)
