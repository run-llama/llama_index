import os
import random
import sys
import textwrap

import ollama
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

# src should sit in the same level as /streamlit_app
curr_dir = os.getcwd()

parent_dir = os.path.dirname(curr_dir)
# parent_dir = os.path.dirname(curr_dir) + '/memary' #Use this if error: src not found. Also move the '/streamlit_app/data' folder to the 'memary' folder, outside the 'src' folder.

sys.path.append(parent_dir + "/src")

from llama_index.packs.memary.base import MemaryChatAgentPack

load_dotenv()

system_persona_txt = "data/system_persona.txt"
user_persona_txt = "data/user_persona.txt"
past_chat_json = "data/past_chat.json"
memory_stream_json = "data/memory_stream.json"
entity_knowledge_store_json = "data/entity_knowledge_store.json"
chat_agent = MemaryChatAgentPack(
    "Personal Agent",
    memory_stream_json,
    entity_knowledge_store_json,
    system_persona_txt,
    user_persona_txt,
    past_chat_json,
)


def create_graph(nodes, edges):
    g = Network(
        notebook=True,
        directed=True,
        cdn_resources="in_line",
        height="500px",
        width="100%",
    )

    for node in nodes:
        g.add_node(node, label=node, title=node)
    for edge in edges:
        # assuming only one relationship type per edge
        g.add_edge(edge[0], edge[1], label=edge[2][0])

    g.repulsion(
        node_distance=200,
        central_gravity=0.12,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )
    return g


def fill_graph(nodes, edges, cypher_query):
    entities = []
    with GraphDatabase.driver(
        uri=chat_agent.neo4j_url,
        auth=(chat_agent.neo4j_username, chat_agent.neo4j_password),
    ) as driver:
        with driver.session() as session:
            result = session.run(cypher_query)
            for record in result:
                path = record["p"]
                rels = [rel.type for rel in path.relationships]

                n1_id = record["p"].nodes[0]["id"]
                n2_id = record["p"].nodes[1]["id"]
                nodes.add(n1_id)
                nodes.add(n2_id)
                edges.append((n1_id, n2_id, rels))
                entities.extend([n1_id, n2_id])


def get_models(llm_models, vision_models):
    models = set()
    try:
        ollama_info = ollama.list()
        for e in ollama_info["models"]:
            models.add(e["model"])
        if "llava:latest" in models:
            vision_models.append("llava:latest")
            models.remove("llava:latest")
        llm_models.extend(list(models))
    except:
        print("No Ollama instance detected.")


cypher_query = "MATCH p = (:Entity)-[r]-()  RETURN p, r LIMIT 1000;"
answer = ""
external_response = ""
st.title("memary")

llm_models = ["gpt-3.5-turbo"]
vision_models = ["gpt-4-vision-preview"]
get_models(llm_models, vision_models)

selected_llm_model = st.selectbox(
    "Select an LLM model to use.",
    (model for model in llm_models),
    index=None,
    placeholder="Select LLM Model...",
)
selected_vision_model = st.selectbox(
    "Select a vision model to use.",
    (model for model in vision_models),
    index=None,
    placeholder="Select Vision Model...",
)

if selected_llm_model and selected_vision_model:
    chat_agent = MemaryChatAgentPack(
        "Personal Agent",
        memory_stream_json,
        entity_knowledge_store_json,
        system_persona_txt,
        user_persona_txt,
        past_chat_json,
        selected_llm_model,
        selected_vision_model,
    )

    st.write(" ")
    clear_memory = st.button("Clear Memory DB")
    query = st.text_input("Ask a question")

    tools = st.multiselect(
        "Select tools to include:",
        ["search", "locate", "vision", "stocks"],  # all options available
        [
            "search",
            "locate",
            "vision",
            "stocks",
        ],  # options that are selected by default
    )

    img_url = ""
    if "vision" in tools:
        img_url = st.text_input("URL of image, leave blank if no image to provide")
        if img_url:
            st.image(img_url, caption="Uploaded Image", use_column_width=True)

    generate_clicked = st.button("Generate")
    st.write("")

    if clear_memory:
        # print("Front end received request to clear memory")
        chat_agent.clearMemory()
        st.write("Memory DB cleared")

    if generate_clicked:
        if query == "":
            st.write("Please enter a question")
            st.stop()

        # get tools
        print("tools enabled: ", tools)
        if len(tools) == 0:
            st.write("Please select at least one tool")
            st.stop()

        chat_agent.update_tools(tools)

        if img_url:
            query += "Image URL: " + img_url
        react_response = ""
        rag_response = (
            "There was no information in knowledge_graph to answer your question."
        )
        chat_agent.add_chat("user", query)
        cypher_query = chat_agent.check_KG(query)
        if cypher_query:
            rag_response, entities = chat_agent.get_routing_agent_response(
                query, return_entity=True
            )
            chat_agent.add_chat("system", "ReAct agent: " + rag_response, entities)
        else:
            # get response
            react_response = chat_agent.get_routing_agent_response(query)
            chat_agent.add_chat("system", "ReAct agent: " + react_response)

        answer = chat_agent.get_response()
        st.subheader("Routing Agent Response")
        routing_response = ""
        with open("data/routing_response.txt", "r") as f:
            routing_response = f.read()
        st.text(str(routing_response))

        if cypher_query:
            nodes = set()
            edges = []  # (node1, node2, [relationships])
            fill_graph(nodes, edges, cypher_query)

            st.subheader("Knoweldge Graph")
            st.code("# Current Cypher Used\n" + cypher_query)
            st.write("")
            st.text("Subgraph:")
            graph = create_graph(nodes, edges)
            graph_html = graph.generate_html(f"graph_{random.randint(0, 1000)}.html")
            components.html(graph_html, height=500, scrolling=True)
        else:
            st.subheader("Knowledge Graph")
            st.text("No information found in the knowledge graph")

        st.subheader("Final Response")
        wrapped_text = textwrap.fill(answer, width=80)
        st.text(wrapped_text)

        if len(chat_agent.memory_stream) > 0:
            # Memory Stream
            memory_items = chat_agent.memory_stream.get_memory()
            memory_items_dicts = [item.to_dict() for item in memory_items]
            df = pd.DataFrame(memory_items_dicts)
            st.write("Memory Stream")
            st.dataframe(df)

            # Entity Knowledge Store
            knowledge_memory_items = chat_agent.entity_knowledge_store.get_memory()
            knowledge_memory_items_dicts = [
                item.to_dict() for item in knowledge_memory_items
            ]
            df_knowledge = pd.DataFrame(knowledge_memory_items_dicts)
            st.write("Entity Knowledge Store")
            st.dataframe(df_knowledge)

            # top entities
            top_entities = chat_agent._select_top_entities()
            df_top = pd.DataFrame(top_entities)
            st.write("Top 20 Entities")
            st.dataframe(df_top)
