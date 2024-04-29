from llama_index.llms.openai import OpenAI
from llama_index.core.llms.llm import LLM

from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer

from llama_index.core.tools import BaseTool
from llama_index.core.settings import Settings

from .tool_importer import create_message_spoke_tool, create_function_placeholder
from .spoke_operator import SpokeOperator
from .spoke_parser import SpokeOutputParser
from .sandbox import set_mem_limit, drop_perms


from typing import (
    Sequence,
    List,
    Optional
)


from llama_index.core.agent import ReActAgent

from llama_index.core.base.llms.types import ChatMessage


class Spoke():
    
    def __init__(
        self, 
        tools: Sequence[BaseTool],
        collab_functions: Sequence[str], 
        llm: LLM = None,
        memory: ChatMemoryBuffer = None,
        verbose: bool = True
    ) -> None:
    
        self.tools = tools
        if self.tools:
            self.tool_name = tools[0].metadata.name 
        else:
            self.tool_name = ""

        self.collab_functions = collab_functions
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=[], llm=self.llm
        )
        
        # Set up spoke operator
        self.spoke_operator = SpokeOperator(self.collab_functions)

        # Create a placeholder for each collabortive functionality
        func_placeholders = create_function_placeholder(self.collab_functions)

        # Set the tool and collabortive functionality list
        tool_functionality_list = self.tools + func_placeholders + [create_message_spoke_tool()] 
        
        # Set up the spoke output parser
        self.spoke_output_parser = SpokeOutputParser(functionality_list=self.collab_functions, spoke_operator=self.spoke_operator)
        # Set up the spoke agent    
        self.spoke_agent = ReActAgent.from_tools(tools=tool_functionality_list, llm=self.llm, memory=self.memory, output_parser=self.spoke_output_parser, verbose=verbose)

    
    def chat(
        self, 
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        ):      
        response = self.spoke_agent.chat(query,chat_history=chat_history)
        return response

        
    def run_process(self, child_sock, request, spoke_id, chat_history=None):        
        # Set seccomp and setrlimit 
        set_mem_limit()
        drop_perms()
        
        self.spoke_operator.spoke_id = spoke_id
        self.spoke_operator.child_sock = child_sock
        request = self.spoke_operator.parse_request(request)
        results = self.chat(request, chat_history)
        self.spoke_operator.return_response(str(results)) 
        
             
from llama_index.core.tools import FunctionTool
def add_numbers(x: int, y: int) -> int:
    """
    Adds the two numbers together and returns the result.
    """
    return x + y

if __name__ == '__main__':
    llm = OpenAI(model="gpt-4-turbo", temperature=0.0, additional_kwargs={"seed": 0})
    function_tool = FunctionTool.from_defaults(fn=add_numbers)
    print(function_tool.metadata)
    print(function_tool.metadata.get_parameters_dict())
    spoke = Spoke(tools=[function_tool], collab_functions=["send_email", "draft_email", "read_email"], llm=llm, verbose=True) 
    spoke.chat("send a email to yuhao.wu@email.com, subject: hello, body: hello world") 
