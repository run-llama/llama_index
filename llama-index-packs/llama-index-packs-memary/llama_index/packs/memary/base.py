import logging
from typing import Optional, List

from memary.agent.base_agent import Agent


class MemaryChatAgentPack(Agent):
    """ChatAgent currently able to support Llama3 running on Ollama (default) and gpt-3.5-turbo for llm models,
    and LLaVA running on Ollama (default) and gpt-4-vision-preview for the vision tool.
    """

    def __init__(
        self,
        name,
        memory_stream_json,
        entity_knowledge_store_json,
        system_persona_txt,
        user_persona_txt,
        past_chat_json,
        llm_model_name="llama3",
        vision_model_name="llava",
        include_from_defaults=["search", "locate", "vision", "stocks"],
    ) -> None:
        super().__init__(
            name,
            memory_stream_json,
            entity_knowledge_store_json,
            system_persona_txt,
            user_persona_txt,
            past_chat_json,
            llm_model_name,
            vision_model_name,
            include_from_defaults,
        )

    def add_chat(self, role: str, content: str, entities: Optional[List[str]] = None):
        """Add a chat to the agent's memory.

        Args:
            role (str): 'system' or 'user'
            content (str): content of the chat
            entities (Optional[List[str]], optional): entities from Memory systems. Defaults to None.
        """
        # Add a chat to the agent's memory.
        self._add_contexts_to_llm_message(role, content)

        if entities:
            self.memory_stream.add_memory(entities)
            self.memory_stream.save_memory()
            self.entity_knowledge_store.add_memory(self.memory_stream.get_memory())
            self.entity_knowledge_store.save_memory()

        self._replace_memory_from_llm_message()
        self._replace_eks_to_from_message()

    def get_chat(self):
        return self.contexts

    def clearMemory(self):
        """Clears Neo4j database and memory stream/entity knowledge store."""
        logging.info("Deleting memory stream and entity knowledge store...")
        self.memory_stream.clear_memory()
        self.entity_knowledge_store.clear_memory()

        logging.info("Deleting nodes from Neo4j...")
        try:
            self.graph_store.query("MATCH (n) DETACH DELETE n")
        except Exception as e:
            logging.error(f"Error deleting nodes: {e}")
        logging.info("Nodes deleted from Neo4j.")

    def _replace_memory_from_llm_message(self):
        """Replace the memory_stream from the llm_message."""
        self.message.llm_message["memory_stream"] = self.memory_stream.get_memory()

    def _replace_eks_to_from_message(self):
        """Replace the entity knowledge store from the llm_message. eks means entity knowledge store."""
        self.message.llm_message[
            "knowledge_entity_store"
        ] = self.entity_knowledge_store.get_memory()
