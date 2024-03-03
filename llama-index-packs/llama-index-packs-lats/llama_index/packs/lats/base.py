from llama_index.core.agent import (
    CustomSimpleAgentWorker,
    Task,
    AgentChatResponse,
)
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.agent import ReActChatFormatter, ReActOutputParser
from llama_index.core.query_pipeline import QueryPipeline as QP, ToolRunnerComponent
from llama_index.llms.openai import OpenAI
from typing import List, Any, Dict, Tuple, Optional, cast
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.prompts import PromptTemplate



class ExpandedTasks(BaseModel):
    """Expanded queries."""

    queries: List[str]



class Node(BaseModel):
    """Node."""
    
    chat_history: List[ChatMessage] = Field(..., description="Chat history.")
    parent: Optional["Node"] = Field(None, description="Parent node.")
    children: List["Node"] = Field(default_factory=list, description="Children nodes.") 
    score: float = Field(0.0, description="Score of the node.")



# taken from the paper
DEFAULT_REFLECTION_PROMPT_STR = """\
Given a query and a conversation trajectory, evaluate two things regarding whether the conversation answers the question: 
- **correctness**
- **completeness**
Provide your reasoning and analysis in detail. 
Focus on the latest thought, action, and observation.
Incomplete trajectories can be correct if the thoughts and actions so far are correct, \
even if the answer is not found yet. 
Do not generate additional thoughts or actions. 


Query: {query}
Conversation History:
{conversation_history}

Please output your reasoning in the following format.

"""

DEFAULT_REFLECTION_PROMPT = PromptTemplate(
    prompt_template_str=DEFAULT_REFLECTION_PROMPT_STR,
)


class Reflection(BaseModel):
    """Reflection."""

    score: float = Field(0.0, description="Score of the reflection indicating **correctness**.")
    is_done: bool = Field(False, description="Is the task done (**completeness**).")
    reasoning: str = Field(..., description="Reasoning about the trajectory")
    


class LATSAgentWorker(CustomSimpleAgentWorker):
    """Agent worker that performs a step of Language Agent Tree Search.

    Source paper: https://arxiv.org/pdf/2310.04406v2.pdf.

    Continues iterating until there's no errors / task is done.

    """

    tools: List[BaseTool] = Field(..., description="List of tools to use.")
    num_expansions: int = Field(5, description="Number of expansions to do.")
    llm: LLM = Field(..., description="LLM to use.")
    reflection_prompt: PromptTemplate = Field(..., description="Reflection prompt.")

    def __init__(
        self, 
        tools: List[BaseTool], 
        llm: Optional[LLM] = None,
        num_expansions: int = 5,
        reflection_prompt: Optional[PromptTemplate] = None,
        **kwargs: Any
    ) -> None:
        """Init params."""
        # validate that all tools are query engine tools
        llm = llm or OpenAI(model="gpt-4", temperature=0.5)
        super().__init__(
            tools=tools,
            llm=llm,
            num_expansions=num_expansions,
            reflection_prompt=reflection_prompt or DEFAULT_REFLECTION_PROMPT,
            **kwargs,
        )

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        return {"count": 0, "current_reasoning": []}

    def _run_candidates(self, node: Node, task: Task, state: Dict[str, Any]) -> List[Dict]:
        """Expand node."""

        # task_program = LLMTextCompletionProgram.from_defaults(
        #     output_cls=ExpandedTasks,
        #     prompt_template_str=DEFAULT_EXPAND_PROMPT_TMPL,
        #     verbose=True,
        # )

        candidate_dicts = []
        tool_runner_component = ToolRunnerComponent(
            self.tools, callback_manager=task.callback_manager
        )
        for i in range(self.num_expansions):
            candidate_dict = {}
            chat_formatter = ReActChatFormatter()
            formatted_prompt = chat_formatter.format(
                self.tools,
                chat_history=task.memory.get(),
                # current_reasoning=state["current_reasoning"],
                current_reasoning=node.chat_history,
            )
            response = self.llm.chat(formatted_prompt)
            output_parser = ReActOutputParser()
            reasoning_step = output_parser.parse(response.message.content)
            if reasoning_step.is_done:
                reasoning_step = cast(ResponseReasoningStep, reasoning_step)
                candidate_dict["current_reasoning"] = [reasoning_step]
            else:
                reasoning_step = cast(ActionReasoningStep, reasoning_step)
                tool_output = tool_runner_component.run_component(
                    tool_name=reasoning_step.action,
                    tool_input=reasoning_step.action_input
                )
                observation_step = ObservationReasoningStep(observation=str(tool_output))
                candidate_dict["current_reasoning"] = [reasoning_step, observation_step]

            candidate_dicts.append(candidate_dict)

        return candidate_dicts

    def _reflect(
        self,
        input: str,
        candidate_dict: Dict,
        state: Dict[str, Any]
    ) -> float:
        """Reflect."""
        history_str = "\n".join([s.get_content() for s in state["current_reasoning"]])
        reflection = self.llm.structured_predict(
            Reflection,
            prompt=self.reflection_prompt,
            query=input,
            conversation_history=history_str,
        )
        return reflection

    def _run_step(
        self, state: Dict[str, Any], task: Task
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        root_node = state["root_node"]
        cur_node = root_node.get_best_node()

        # expand the given node, generate n candidates
        # for each candidate, run tool, get response
        candidate_dicts = self._run_candidates(cur_node, task.input, state)

        # run all the candidates 
        for candidate_dict in candidate_dicts:
            # either do rollout or estimate reward through self reflection
            # TODO: currently we do the latter
            reflection = self._reflect(candidate_dict, state)
            
            # update the state by adding new nodes, and backpropagating the reward
            # to the parents
            state = self._update_state(candidate, candidate_response, reward, state)
        

        # # pick the next best node to expand
        # next_node = self._pick_next_node(state, candidates)

        # return response
        return AgentChatResponse(response=str(response)), is_done

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
        # nothing to finalize here
        # this is usually if you want to modify any sort of
        # internal state beyond what is set in `_initialize_state`
        pass


