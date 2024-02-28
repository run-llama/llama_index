from typing import Any, Dict, Optional

from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_pipeline import QueryPipeline, InputComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

_REASONING_MODULES = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternative perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "16. What is the core issue or problem that needs to be addressed?",
    "17. What are the underlying causes or factors contributing to the problem?",
    "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "19. What are the potential obstacles or challenges that might arise in solving this problem?",
    "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "23. How can progress or success in solving the problem be measured or evaluated?",
    "24. What indicators or metrics can be used?",
    "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "30. Is the problem a design challenge that requires creative solutions and innovation?",
    "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "33. What kinds of solution typically are produced for this kind of problem specification?",
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    "38. Let’s think step by step ."
    "39. Let’s make a step by step plan and implement it with good notation and explanation.",
]

_REASONING_MODULES = "\n".join(_REASONING_MODULES)


class PipelineConfigurator:
    """Configures and sets up a query pipeline for a given task and reasoning modules."""

    def __init__(self, task, reasoning_modules, verbose, llm) -> None:
        """Initializes the configurator with task details, reasoning modules, and verbosity setting."""
        self.task = task
        self.reasoning_modules = reasoning_modules
        self.verbose = verbose
        self.pipeline = QueryPipeline(verbose=self.verbose)
        self.llm = llm

    def setup_templates(self) -> None:
        """Sets up prompt templates for different stages of the pipeline."""
        self.select_prompt_template = PromptTemplate(
            "Given the task: {task}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n {reasoning_modules}"
        )
        self.adapt_prompt_template = PromptTemplate(
            "Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task}"
        )
        self.implement_prompt_template = PromptTemplate(
            "Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task}"
        )
        self.reasoning_prompt_template = PromptTemplate(
            "Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task}"
        )

    def add_modules(self) -> None:
        """Adds necessary modules and their configurations to the pipeline."""
        self.pipeline.add_modules(
            {
                "input": InputComponent(),
                "select_llm": self.llm,
                "adapt_llm": self.llm,
                "implement_llm": self.llm,
                "reasoning_llm": self.llm,
                "select_prompt_template": self.select_prompt_template,
                "adapt_prompt_template": self.adapt_prompt_template,
                "implement_prompt_template": self.implement_prompt_template,
                "reasoning_prompt_template": self.reasoning_prompt_template,
            }
        )

    def setup_links(self) -> None:
        """Defines the connections (links) between the different pipeline modules."""
        # STAGE-1: SELECT subset of reasoning Modules.
        self.pipeline.add_link(
            "input", "select_prompt_template", src_key="task", dest_key="task"
        )
        self.pipeline.add_link(
            "input",
            "select_prompt_template",
            src_key="reasoning_modules",
            dest_key="reasoning_modules",
        )
        self.pipeline.add_link("select_prompt_template", "select_llm")

        # STAGE-1: ADAPT selected reasoning modules to the task.
        self.pipeline.add_link(
            "select_llm", "adapt_prompt_template", dest_key="selected_modules"
        )
        self.pipeline.add_link(
            "input", "adapt_prompt_template", src_key="task", dest_key="task"
        )
        self.pipeline.add_link("adapt_prompt_template", "adapt_llm")

        # STAGE-1: IMPLEMENT provides reasoning structure for the task.
        self.pipeline.add_link(
            "adapt_llm", "implement_prompt_template", dest_key="adapted_modules"
        )
        self.pipeline.add_link(
            "input", "implement_prompt_template", src_key="task", dest_key="task"
        )
        self.pipeline.add_link("implement_prompt_template", "implement_llm")

        # STAGE-2: Uses the generated reasoning structure for the task to generate an answer.
        self.pipeline.add_link(
            "implement_llm", "reasoning_prompt_template", dest_key="reasoning_structure"
        )
        self.pipeline.add_link(
            "input", "reasoning_prompt_template", src_key="task", dest_key="task"
        )
        self.pipeline.add_link("reasoning_prompt_template", "reasoning_llm")

    def configure(self) -> QueryPipeline:
        """Configures and returns the fully set up pipeline."""
        self.setup_templates()
        self.add_modules()
        self.setup_links()
        return self.pipeline


class SelfDiscoverPack(BaseLlamaPack):
    """Self-Discover Pack."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = True,
    ) -> None:
        """Init params."""
        self.llm = llm or OpenAI(model="gpt-3.5-turbo")
        self.reasoning_modules = _REASONING_MODULES
        self.verbose = verbose

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "reasoning_modules": self.reasoning_modules}

    def run(self, task):
        """Runs the configured pipeline for a specified task and reasoning modules."""
        configurator = PipelineConfigurator(
            task, self.reasoning_modules, self.verbose, self.llm
        )
        pipeline = configurator.configure()
        return pipeline.run(task=task, reasoning_modules=self.reasoning_modules)
