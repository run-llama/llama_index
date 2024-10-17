from typing import Any

from llama_index.llms.oci_genai import OCIGenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput


# Define a real tool for testing purposes
class ExampleTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="example_tool",
            description="An example tool that processes input text.",
            fn_schema=None,  # Use the default schema if you don't have custom parameters
        )

    def __call__(self, input: Any) -> ToolOutput:
        # Implement the processing logic here
        processed_text = f"Processed: {input['input_text']}"
        return ToolOutput(content=processed_text, tool_name=self.metadata.name, raw_input=input, raw_output=None)



# Define the chat messages
messages = [
    ChatMessage(role="user", content="Can you use a tool to process my text?")
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

# Define the tool you want to use
tool = ExampleTool()

# Use chat_with_tools method to invoke the tool within the chat response
response = llm.chat_with_tools(
    tools=[tool],
    user_msg="Can you use a tool to process my text?",
    chat_history=messages,
)

# Print the response
print("Response message:", response.message.content)
