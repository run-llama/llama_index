DEFAULT_HANDOFF_PROMPT = """Useful for handing off to another agent.
If you are currently not equipped to handle the user's request, or another agent is better suited to handle the request, please hand off to the appropriate agent.

Currently available agents:
{agent_info}
"""

DEFAULT_STATE_PROMPT = """Current state:
{state}

Current message:
{msg}
"""

DEFAULT_HANDOFF_OUTPUT_PROMPT = "Agent {to_agent} is now handling the request due to the following reason: {reason}.\nPlease continue with the current request."

DEFAULT_EARLY_STOPPING_PROMPT = """You have reached the maximum number of iterations ({max_iterations}).
Based on the information gathered so far, please provide a helpful final response to the user's original query.
Do not attempt to use any more tools. Simply summarize what you have learned and provide the best possible answer."""
