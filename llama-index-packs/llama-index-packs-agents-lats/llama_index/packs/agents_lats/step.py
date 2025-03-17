import asyncio
from typing import List, Any, Dict, Tuple, Optional, cast

from llama_index.core.agent import (
    AgentChatResponse,
    CustomSimpleAgentWorker,
    ReActChatFormatter,
    ReActOutputParser,
    Task,
)
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolSelection, acall_tool_with_selection
from llama_index.core.utils import print_text

from llama_index.packs.agents_lats.types import (
    DEFAULT_CANDIDATES_PROMPT,
    DEFAULT_REFLECTION_PROMPT,
    Candidates,
    Evaluation,
    SearchNode,
)


class LATSAgentWorker(CustomSimpleAgentWorker):
    """
    Agent worker that performs a step of Language Agent Tree Search.

    Source paper: https://arxiv.org/pdf/2310.04406v2.pdf.

    Continues iterating until there's no errors / task is done.

    """

    num_expansions: int = Field(default=2, description="Number of expansions to do.")
    reflection_prompt: PromptTemplate = Field(..., description="Reflection prompt.")
    candiate_expansion_prompt: PromptTemplate = Field(
        ..., description="Candidate expansion prompt."
    )
    max_rollouts: int = Field(
        default=5,
        description=(
            "Max rollouts. By default, -1 means that we keep going until the first solution is found."
        ),
    )

    chat_formatter: ReActChatFormatter = Field(
        default_factory=ReActChatFormatter, description="Chat formatter."
    )

    def __init__(
        self,
        tools: List[BaseTool],
        llm: Optional[LLM] = None,
        num_expansions: int = 2,
        max_rollouts: int = 5,
        reflection_prompt: Optional[PromptTemplate] = None,
        candiate_expansion_prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        # validate that all tools are query engine tools
        llm = llm or Settings.llm
        super().__init__(
            tools=tools,
            llm=llm,
            num_expansions=num_expansions,
            max_rollouts=max_rollouts,
            reflection_prompt=reflection_prompt or DEFAULT_REFLECTION_PROMPT,
            candiate_expansion_prompt=candiate_expansion_prompt
            or DEFAULT_CANDIDATES_PROMPT,
            **kwargs,
        )

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        # initialize root node
        root_node = SearchNode(
            current_reasoning=[ObservationReasoningStep(observation=task.input)],
            evaluation=Evaluation(score=1),  # evaluation for root node is blank
        )
        return {"count": 0, "solution_queue": [], "root_node": root_node}

    async def _arun_candidate(
        self,
        node: SearchNode,
        task: Task,
    ) -> List[BaseReasoningStep]:
        """
        Generate candidate for a given node.

        Generically we sample the action space to generate new candidate nodes.

        Practically since we're using a ReAct powered agent, this means
        using generating a ReAct trajectory, running a tool.

        """
        output_parser = ReActOutputParser()
        # format react prompt
        formatted_prompt = self.chat_formatter.format(
            self.tools,
            chat_history=task.memory.get(),
            current_reasoning=node.current_reasoning,
        )
        # run LLM
        response = await self.llm.achat(formatted_prompt)
        # parse output into reasoning step

        try:
            reasoning_step = output_parser.parse(response.message.content)
        except ValueError as e:
            reasoning_step = ResponseReasoningStep(
                thought=response.message.content,
                response=f"Encountered an error parsing: {e!s}",
            )
        # get response or run tool
        if reasoning_step.is_done:
            reasoning_step = cast(ResponseReasoningStep, reasoning_step)
            current_reasoning = [reasoning_step]
        else:
            reasoning_step = cast(ActionReasoningStep, reasoning_step)
            tool_selection = ToolSelection(
                tool_id=reasoning_step.action,
                tool_name=reasoning_step.action,
                tool_kwargs=reasoning_step.action_input,
            )
            try:
                tool_output = await acall_tool_with_selection(
                    tool_selection, self.tools, verbose=self.verbose
                )
            except Exception as e:
                tool_output = f"Encountered error: {e!s}"
            observation_step = ObservationReasoningStep(observation=str(tool_output))
            current_reasoning = [reasoning_step, observation_step]

        return current_reasoning

    async def _aevaluate(
        self,
        cur_node: SearchNode,
        current_reasoning: List[BaseReasoningStep],
        input: str,
    ) -> float:
        """Evaluate."""
        all_reasoning = cur_node.current_reasoning + current_reasoning
        history_str = "\n".join([s.get_content() for s in all_reasoning])
        evaluation = await self.llm.astructured_predict(
            Evaluation,
            prompt=self.reflection_prompt,
            query=input,
            conversation_history=history_str,
        )
        if self.verbose:
            print_text(
                f"> Evaluation for input {input}\n: {evaluation}\n\n", color="pink"
            )

        return evaluation

    async def _get_next_candidates(
        self,
        cur_node: SearchNode,
        input: str,
    ) -> List[str]:
        """Get next candidates."""
        # get candidates
        history_str = "\n".join([s.get_content() for s in cur_node.current_reasoning])

        candidates = await self.llm.astructured_predict(
            Candidates,
            prompt=self.candiate_expansion_prompt,
            query=input,
            conversation_history=history_str,
            num_candidates=self.num_expansions,
        )
        candidate_strs = candidates.candidates[: self.num_expansions]
        if self.verbose:
            print_text(f"> Got candidates: {candidate_strs}\n", color="yellow")

        # ensure we have the right number of candidates
        if len(candidate_strs) < self.num_expansions:
            return (candidate_strs * self.num_expansions)[: self.num_expansions]
        else:
            return candidate_strs[: self.num_expansions]

    def _update_state(
        self,
        node: SearchNode,
        current_reasoning: List[BaseReasoningStep],
        evaluation: Evaluation,
    ) -> SearchNode:
        """Update state."""
        # create child node
        new_node = SearchNode(
            current_reasoning=node.current_reasoning + current_reasoning,
            parent=node,
            children=[],
            evaluation=evaluation,
        )
        node.children.append(new_node)

        # backpropagate the reward
        new_node.backpropagate(evaluation.score)

        return new_node

    def _run_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:
        """
        Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        return asyncio.run(self._arun_step(state, task, input))

    async def _arun_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:
        """
        Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        root_node = state["root_node"]
        cur_node = root_node.get_best_leaf()
        if self.verbose:
            print_text(
                f"> Selecting node to expand: {cur_node.answer}\n", color="green"
            )

        # expand the given node, generate n candidate nodes
        new_candidates = await self._get_next_candidates(
            cur_node,
            task.input,
        )

        new_nodes = []
        for candidate in new_candidates:
            new_nodes.append(
                self._update_state(
                    cur_node,
                    [ObservationReasoningStep(observation=candidate)],
                    Evaluation(score=1),  # evaluation for candidate node is blank
                )
            )

        # expand the given node, generate n candidates
        # for each candidate, run tool, get response

        solution_queue: List[SearchNode] = state["solution_queue"]

        # first, generate the candidates
        candidate_jobs = [
            self._arun_candidate(new_node, task) for new_node in new_nodes
        ]
        all_new_reasoning_steps = await asyncio.gather(*candidate_jobs)
        if self.verbose:
            for new_reasoning_steps in all_new_reasoning_steps:
                out_txt = "\n".join([s.get_content() for s in new_reasoning_steps])
                print_text(f"> Generated new reasoning step: {out_txt}\n", color="blue")
        # then, evaluate the candidates
        eval_jobs = [
            self._aevaluate(new_node, new_reasoning_steps, task.input)
            for new_node, new_reasoning_steps in zip(new_nodes, all_new_reasoning_steps)
        ]
        evaluations = await asyncio.gather(*eval_jobs)
        # then, update the state
        for new_reasoning_steps, cur_new_node, evaluation in zip(
            all_new_reasoning_steps, new_nodes, evaluations
        ):
            new_node = self._update_state(cur_new_node, new_reasoning_steps, evaluation)
            if new_node.is_done:
                if self.verbose:
                    print_text(
                        f"> Found solution node: {new_node.answer}\n", color="cyan"
                    )
                solution_queue.append(new_node)

        # check if done
        state["count"] += 1
        if self.max_rollouts == -1 and solution_queue:
            is_done = True
        elif self.max_rollouts > 0 and state["count"] >= self.max_rollouts:
            is_done = True
        else:
            is_done = False

        # determine response
        if solution_queue:
            best_solution_node = max(solution_queue, key=lambda x: x.score)
            response = best_solution_node.answer
        else:
            response = "I am still thinking."

        if self.verbose:
            print_text(f"> Got final response: {response!s}\n", color="green")

        # return response
        return AgentChatResponse(response=str(response)), is_done

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
        # nothing to finalize here
        # this is usually if you want to modify any sort of
        # internal state beyond what is set in `_initialize_state`
