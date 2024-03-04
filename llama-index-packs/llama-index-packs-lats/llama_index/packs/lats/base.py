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
from llama_index.core.agent import AgentRunner
import math
from llama_index.core.llama_pack import BaseLlamaPack



class SearchNode(BaseModel):
    """Search node.

    Named differently from `Node` which is a core module in LlamaIndex.
    
    """
    
    current_reasoning: List[BaseReasoningStep] = Field(..., description="Current reasoning.")
    parent: Optional["SearchNode"] = Field(None, description="Parent node.")
    children: List["SearchNode"] = Field(default_factory=list, description="Children nodes.") 
    evaluation: "Evaluation" = Field(0.0, description="Evaluation of the node.")
    visits: int = Field(0, description="Number of visits to the node.")
    
    @property
    def answer(self) -> Optional[str]:
        """Answer."""
        if not self.current_reasoning:
            return None
        return self.current_reasoning[-1].content

    @property
    def is_done(self) -> bool:
        """Is the node done."""
        return self.evaluation.is_done

    @property
    def upper_confidence_bound(self) -> float:
        """Upper confidence bound."""
        return self.score + 1.0 * math.sqrt(math.log(self.parent.visits) / self.visits)

    def backpropagate(self, reward: float) -> None:
        """Backpropagate the reward."""
        self.visits += 1
        cur_node = self
        while cur_node is not None:
            cur_node.score = (reward + (cur_node.visits - 1) * cur_node.score) / self.visits
            cur_node = cur_node.parent

    def get_best_leaf(self) -> "SearchNode":
        """Get best leaf node.

        Get best leaf node across any children nodes.
        
        """

        # only get children that aren't done yet
        free_children = [c for c in self.children if not c.is_done]
        if not free_children:
            return self

        best_child = max(free_children, key=lambda x: x.upper_confidence_bound)
        return best_child.get_best_leaf()
        


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


class Evaluation(BaseModel):
    """Evaluation of a given node.

    Currently evaluation is done by using LLMs to reflect on the trajectory.
    
    """

    score: int = Field(default=0, description="Score of the reflection indicating **correctness**. Integer from 1-10", lte=10, gte=0)
    is_done: bool = Field(False, description="Is the task done (**completeness**).")
    reasoning: str = Field(default="", description="Reasoning about the trajectory")
    


class LATSAgentWorker(CustomSimpleAgentWorker):
    """Agent worker that performs a step of Language Agent Tree Search.

    Source paper: https://arxiv.org/pdf/2310.04406v2.pdf.

    Continues iterating until there's no errors / task is done.

    """

    tools: List[BaseTool] = Field(..., description="List of tools to use.")
    num_expansions: int = Field(5, description="Number of expansions to do.")
    llm: LLM = Field(..., description="LLM to use.")
    reflection_prompt: PromptTemplate = Field(..., description="Reflection prompt.")
    max_depth: int = Field(5, description="Max depth of the tree.")
    max_rollouts: int = Field(
        -1, description=(
            "Max rollouts. By default, -1 means that we keep going until the first solution is found."
        )
    )
    
    chat_formatter: ReActChatFormatter = Field(
        default_factory=ReActChatFormatter, description="Chat formatter."
    )

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
        # initialize root node
        root_node = SearchNode(
            current_reasoning=[ObservationReasoningStep(observation=task.input)],
            evaluation=Evaluation(),  # evaluation for root node is blank
        )
        return {"count": 0, "solution_queue": [], "root_node": root_node}

    def _run_candidate(
        self, node: SearchNode, task: Task, state: Dict[str, Any]
    ) -> List[BaseReasoningStep]:
        """Generate candidate for a given node.

        Generically we sample the action space to generate new candidate nodes.

        Practically since we're using a ReAct powered agent, this means
        using generating a ReAct trajectory, running a tool.
        
        """

        tool_runner_component = ToolRunnerComponent(
            self.tools, callback_manager=task.callback_manager
        )
        output_parser = ReActOutputParser()
        # format react prompt 
        formatted_prompt = self.chat_formatter.format(
            self.tools,
            chat_history=task.memory.get(),
            current_reasoning=node.current_reasoning,
        )
        # run LLM
        response = self.llm.chat(formatted_prompt)
        # parse output into reasoning step
        reasoning_step = output_parser.parse(response.message.content)
        # get response or run tool
        if reasoning_step.is_done:
            reasoning_step = cast(ResponseReasoningStep, reasoning_step)
            current_reasoning = [reasoning_step]
        else:
            reasoning_step = cast(ActionReasoningStep, reasoning_step)
            tool_output = tool_runner_component.run_component(
                tool_name=reasoning_step.action,
                tool_input=reasoning_step.action_input
            )
            observation_step = ObservationReasoningStep(observation=str(tool_output))
            current_reasoning = [reasoning_step, observation_step]

        return current_reasoning

    def _evaluate(
        self,
        input: str,
        candidate_dict: Dict,
        state: Dict[str, Any]
    ) -> float:
        """Reflect."""
        history_str = "\n".join([s.get_content() for s in state["current_reasoning"]])
        evaluation = self.llm.structured_predict(
            Evaluation,
            prompt=self.reflection_prompt,
            query=input,
            conversation_history=history_str,
        )

        return evaluation

    def _update_state(
        self, 
        node: SearchNode,
        current_reasoning: List[BaseReasoningStep],
        reflection: Evaluation,
        state: Dict[str, Any]
    ) -> None:
        """Update state."""
        # create child node
        new_node = SearchNode(
            current_reasoning=node.current_reasoning + current_reasoning,
            parent=node,
            children=[],
            score=reflection.score,
        )
        node.children.append(new_node)
        
        # backpropagate the reward
        new_node.backpropagate(reflection.score)
        
        # # backpropagate the reward
        # while node is not None:
        #     node.score += reflection.score
        #     node = node.parent
        

    def _run_step(
        self, state: Dict[str, Any], task: Task
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        root_node = state["root_node"]
        cur_node = root_node.get_best_leaf()

        # expand the given node, generate n candidates
        # for each candidate, run tool, get response

        solution_queue: List[SearchNode] = state["solution_queue"]

        
        # first, generate the candidates
        new_reasoning_steps = [
            self._run_candidate(cur_node, task.input, state)
            for _ in enumerate(range(self.num_expansions))
        ]
        # then, evaluate the candidates
        evaluations = [
            self._evaluate(new_reasoning_steps, state)
            for new_reasoning_steps in new_reasoning_steps
        ]
        # then, update the state
        for new_reasoning_steps, evaluation in zip(new_reasoning_steps, evaluations):
            new_node = self._update_state(cur_node, new_reasoning_steps, evaluation, state)
            if new_node.is_done:
                solution_queue.append(new_node)
        

        # check if done
        state["count"] += 1
        if self.max_rollouts == -1 and solution_queue:
            is_done = True
        elif state["count"] >= self.max_rollouts:
            is_done = True
        else:
            is_done = False

        # determine response
        if solution_queue:
            best_solution_node = max(solution_queue, key=lambda x: x.score)
            response = best_solution_node.answer
        else:
            response = "I am still thinking."

        # return response
        return AgentChatResponse(response=str(response)), is_done

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
        # nothing to finalize here
        # this is usually if you want to modify any sort of
        # internal state beyond what is set in `_initialize_state`
        pass



class LATSPack(BaseLlamaPack):
    """Pack for running the LATS agent."""

    def __init__(
        self,
        **kwargs: Any
    ) -> None:
        """Init params."""
        agent_worker = LATSAgentWorker(**kwargs)
        agent = AgentRunner(agent_worker)
        self.agent_worker = agent_worker

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "agent_worker": self.agent_worker,
            "agent": self.agent,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run."""
        return self.agent.chat(*args, **kwargs)