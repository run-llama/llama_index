"""Streamlit UI for manual testing of the LlamaIndex Merge Agent Handler ToolSpec."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from llama_index.tools.merge_agent_handler import MergeAgentHandlerToolSpec


def _render_output(raw: Any) -> None:
    if isinstance(raw, (dict, list)):
        st.json(raw)
        return
    if isinstance(raw, str):
        try:
            st.json(json.loads(raw))
            return
        except json.JSONDecodeError:
            st.code(raw, language="text")
            return
    st.write(raw)


def _build_spec(
    api_key: str,
    tool_pack_id: str,
    registered_user_id: str,
    environment: str,
) -> MergeAgentHandlerToolSpec:
    return MergeAgentHandlerToolSpec(
        api_key=api_key,
        tool_pack_id=tool_pack_id or None,
        registered_user_id=registered_user_id or None,
        environment=environment,
    )


st.set_page_config(page_title="LlamaIndex Merge Agent Handler", page_icon=":toolbox:", layout="wide")
st.title("LlamaIndex Merge Agent Handler Tester")
st.caption("Use this UI to test list_tool_packs, list_registered_users, list_tools, and call_tool.")

with st.sidebar:
    st.header("Connection")
    api_key_input = st.text_input("Merge API Key", type="password")
    tool_pack_id_input = st.text_input("Default Tool Pack ID (optional)")
    registered_user_id_input = st.text_input("Default Registered User ID (optional)")
    environment_input = st.selectbox("Environment", ["production", "test"], index=0)
    initialize = st.button("Initialize ToolSpec", type="primary")

if initialize:
    if not api_key_input:
        st.error("Merge API Key is required to initialize the tool spec.")
    else:
        try:
            existing = st.session_state.get("tool_spec")
            if existing:
                existing.close()
            st.session_state["tool_spec"] = _build_spec(
                api_key=api_key_input,
                tool_pack_id=tool_pack_id_input,
                registered_user_id=registered_user_id_input,
                environment=environment_input,
            )
            st.success("ToolSpec initialized.")
        except Exception as exc:
            st.error(f"Failed to initialize tool spec: {exc}")

tool_spec = st.session_state.get("tool_spec")
if not tool_spec:
    st.info("Initialize the tool spec from the sidebar to begin testing.")
    st.stop()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Discovery")

    if st.button("List Tool Packs"):
        try:
            _render_output(tool_spec.list_tool_packs())
        except Exception as exc:
            st.error(str(exc))

    if st.button("List Registered Users"):
        try:
            _render_output(tool_spec.list_registered_users(environment_input))
        except Exception as exc:
            st.error(str(exc))

with right_col:
    st.subheader("MCP Tools")
    list_tool_pack_id = st.text_input("Tool Pack ID override", value=tool_pack_id_input, key="list_tool_pack")
    list_registered_user_id = st.text_input(
        "Registered User ID override",
        value=registered_user_id_input,
        key="list_registered_user",
    )

    if st.button("List MCP Tools"):
        try:
            _render_output(tool_spec.list_tools(list_tool_pack_id or None, list_registered_user_id or None))
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.subheader("Call MCP Tool")
call_tool_name = st.text_input("Tool Name")
call_tool_pack_id = st.text_input(
    "Tool Pack ID override (optional)",
    value=tool_pack_id_input,
    key="call_tool_pack",
)
call_registered_user_id = st.text_input(
    "Registered User ID override (optional)",
    value=registered_user_id_input,
    key="call_registered_user",
)
call_arguments = st.text_area("Arguments JSON", value="{}", height=180)

if st.button("Call Tool", type="primary"):
    if not call_tool_name:
        st.error("Tool Name is required.")
    else:
        _render_output(
            tool_spec.call_tool(
                tool_name=call_tool_name,
                arguments=call_arguments,
                tool_pack_id=call_tool_pack_id or None,
                registered_user_id=call_registered_user_id or None,
            )
        )
