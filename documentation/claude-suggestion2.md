Deep Dive: Issue #18412 - Structured Output for AgentWorkflow

Issue Summary

Requester: Pedro Azevedo (@pazevedo-hyland)
Priority: P0 (Highest)
Status: Open since April 9, 2025
Maintainer Response: Logan Markewich acknowledged the complexity

---

The Problem

Users want AgentWorkflow to return structured output (Pydantic models) during tool-calling workflows, not just after.
Currently:

# What users want:

workflow = AgentWorkflow(agents=[...], output_cls=MyStructuredOutput)
result = await workflow.run("Calculate 30 \* 60")

# result.structured_response should be a validated Pydantic model

The Catch: This is "pretty hard" (per maintainer) because you're combining:

1. Tool usage (which uses structured function calling)
2. Structured output (which is basically also a tool/function call)

---

Current Implementation Analysis

Looking at the code, there IS existing structured output support, but with significant limitations:

Current Approach (Lines 576-610 in multi_agent_workflow.py):

# Option 1: Custom function

if self.structured_output_fn is not None:
output.structured_response = await self.structured_output_fn(messages)

# Option 2: Extra LLM call using output_cls

if self.output_cls is not None:
output.structured_response = await generate_structured_response(
messages=llm_input, llm=agent.llm, output_cls=self.output_cls
)

The generate_structured_response utility (utils.py:35-42):

async def generate_structured_response(
messages: List[ChatMessage], llm: LLM, output_cls: Type[BaseModel]
) -> Dict[str, Any]:
xml_message = messages_to_xml_format(messages)
structured_response = await llm.as_structured_llm(output_cls).achat(messages=xml_message)
return json.loads(structured_response.message.content)

Key Limitations:
┌───────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
│ Issue │ Impact │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Extra LLM call │ Adds latency + cost after every agent completion │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Converts to XML first │ Loses context/format, may confuse LLM │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Generic approach │ Doesn't leverage native provider APIs (OpenAI's response_format, Gemini's json_mode) │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ No streaming │ Structured output doesn't stream incrementally │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Separate from tools │ Can't combine tool_choice + structured output in single request │
└───────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

---

Maintainer's Suggested Solutions

From the comments:

Option 1: Make the structured output a tool, set tool_choice to required

- Caveat: Need to ensure all APIs support tool_choice: required

Option 2: Make an extra LLM call before returning to get the output

- Works for any LLM, but adds significant latency

User workaround (from @pazevedo-hyland):
"I've added structured output as a tool + some prompt engineering and it solved it."

---

Architectural Analysis for Your Solution

This is a cross-cutting architectural problem. Here's the component map:

┌──────────────────────────────────────────────────────────────────┐
│ AgentWorkflow │
│ ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│ │ FunctionAgent │ │ ReActAgent │ │ CodeActAgent │ │
│ └────────┬────────┘ └────────┬────────┘ └──────┬───────┘ │
│ │ │ │ │
│ └──────────────────────┼─────────────────────┘ │
│ │ │
│ ┌─────────────▼─────────────┐ │
│ │ BaseWorkflowAgent │ │
│ │ - take_step() │ │
│ │ - parse_agent_output() │◄── Structured │
│ │ - finalize() │ output here │
│ └─────────────┬─────────────┘ │
│ │ │
└──────────────────────────────────┼────────────────────────────────┘
│
┌──────────────▼──────────────┐
│ LLM Providers │
│ - OpenAI (response_format) │
│ - OpenAIResponses API │
│ - Gemini (json_mode) │
│ - Anthropic │
└─────────────────────────────┘

---

Proposed Solution Architecture

Based on my analysis, here's a comprehensive solution:

Option A: "Structured Output as Final Tool" Pattern (Recommended)

class StructuredOutputTool(BaseTool):
"""Auto-generated tool for structured output extraction."""

      def __init__(self, output_cls: Type[BaseModel]):
          self.output_cls = output_cls
          self.metadata = ToolMetadata(
              name="finalize_response",
              description=f"Call this to provide final structured response: {output_cls.model_json_schema()}",
              fn_schema=output_cls,
              return_direct=True,
          )

Implementation changes:

1. When output_cls is set, auto-inject a "finalize" tool
2. Modify agent to recognize when this tool is called
3. Parse the tool arguments as the structured output
4. Works with existing tool_choice: required on final turn

Option B: Native Provider Integration (Higher Impact)

class StructuredOutputConfig:
"""Configuration for structured output."""
output_cls: Type[BaseModel]
mode: Literal["tool", "response_format", "json_mode"] = "auto"

# In AgentWorkflow:

def \_get_structured_output_config(self, llm: LLM) -> dict:
"""Get provider-specific structured output config."""
if llm.supports_response_format: # OpenAI, Gemini
return {"response_format": {"type": "json_schema", ...}}
else:
return self.\_get_tool_based_approach()

---

Complexity Assessment
┌────────────────────────┬─────────────┬───────────────────────────────────────────────┐
│ Aspect │ Difficulty │ Reason │
├────────────────────────┼─────────────┼───────────────────────────────────────────────┤
│ Core Logic │ Medium │ Clear pattern, existing infrastructure │
├────────────────────────┼─────────────┼───────────────────────────────────────────────┤
│ Provider Abstraction │ High │ Each LLM has different structured output APIs │
├────────────────────────┼─────────────┼───────────────────────────────────────────────┤
│ Testing │ Medium-High │ Need mocks for multiple providers │
├────────────────────────┼─────────────┼───────────────────────────────────────────────┤
│ Backward Compatibility │ Low │ Existing params (output_cls) should work │
├────────────────────────┼─────────────┼───────────────────────────────────────────────┤
│ Documentation │ Medium │ Need examples for each approach │
└────────────────────────┴─────────────┴───────────────────────────────────────────────┘
Estimated Scope:

- Core implementation: 300-500 lines
- Tests: 200-400 lines
- Provider-specific: 100-200 lines per provider

---

Files You'll Need to Modify
┌─────────────────────────────────────────────────────────┬─────────────────────────────────────────┐
│ File │ Changes │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
│ llama_index/core/agent/workflow/multi_agent_workflow.py │ Main orchestration logic │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
│ llama_index/core/agent/workflow/base_agent.py │ Base agent structured output handling │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
│ llama_index/core/agent/workflow/function_agent.py │ Tool-based structured output │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
│ llama_index/core/agent/utils.py │ Improve generate_structured_response │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
│ llama_index/core/llms/llm.py │ Add supports_structured_output property │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
│ Provider LLMs (OpenAI, etc.) │ Native response_format support │
└─────────────────────────────────────────────────────────┴─────────────────────────────────────────┘

---

EB-1 Contribution Assessment
┌────────────────────────┬───────┬──────────────────────────────────────────────────┐
│ Criteria │ Score │ Justification │
├────────────────────────┼───────┼──────────────────────────────────────────────────┤
│ Original Contribution │ ★★★★★ │ Solves unsolved problem in popular framework │
├────────────────────────┼───────┼──────────────────────────────────────────────────┤
│ Technical Depth │ ★★★★☆ │ Cross-cutting architecture, provider abstraction │
├────────────────────────┼───────┼──────────────────────────────────────────────────┤
│ Impact │ ★★★★★ │ P0 priority, affects all AgentWorkflow users │
├────────────────────────┼───────┼──────────────────────────────────────────────────┤
│ Visibility │ ★★★★★ │ Core feature in high-profile AI framework │
├────────────────────────┼───────┼──────────────────────────────────────────────────┤
│ Demonstrates Expertise │ ★★★★★ │ Shows solution architect skills │
└────────────────────────┴───────┴──────────────────────────────────────────────────┘
Recommendation: This is an excellent EB-1 contribution. It demonstrates:

- Architectural vision (designing cross-provider abstraction)
- Problem-solving (resolving tool/structured output conflict)
- Impact (P0 issue in popular LLM framework with 40k+ stars)

---

Next Steps

Would you like me to:

1. Draft a detailed implementation plan with code structure?
2. Create a proof-of-concept for the "structured output as tool" approach?
3. Analyze competing approaches (LangChain, Haystack) for inspiration?
4. Prepare a GitHub comment to claim the issue and propose your approach?
