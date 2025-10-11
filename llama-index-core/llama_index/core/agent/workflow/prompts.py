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
