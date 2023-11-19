"""PII postprocessor."""
import json
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts.base import PromptTemplate
from llama_index.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.service_context import ServiceContext

DEFAULT_PII_TMPL = (
    "The current context information is provided. \n"
    "A task is also provided to mask the PII within the context. \n"
    "Return the text, with all PII masked out, and a mapping of the original PII "
    "to the masked PII. \n"
    "Return the output of the task in JSON. \n"
    "Context:\n"
    "Hello Zhang Wei, I am John. "
    "Your AnyCompany Financial Services, "
    "LLC credit card account 1111-0000-1111-0008 "
    "has a minimum payment of $24.53 that is due "
    "by July 31st. Based on your autopay settings, we will withdraw your payment. "
    "Task: Mask out the PII, replace each PII with a tag, and return the text. Return the mapping in JSON. \n"
    "Output: \n"
    "Hello [NAME1], I am [NAME2]. "
    "Your AnyCompany Financial Services, "
    "LLC credit card account [CREDIT_CARD_NUMBER] "
    "has a minimum payment of $24.53 that is due "
    "by [DATE_TIME]. Based on your autopay settings, we will withdraw your payment. "
    "Output Mapping:\n"
    '{{"NAME1": "Zhang Wei", "NAME2": "John", "CREDIT_CARD_NUMBER": "1111-0000-1111-0008", "DATE_TIME": "July 31st"}}\n'
    "Context:\n{context_str}\n"
    "Task: {query_str}\n"
    "Output: \n"
    ""
)


class PIINodePostprocessor(BaseNodePostprocessor):
    """PII Node processor.

    NOTE: the ServiceContext should contain a LOCAL model, not an external API.

    NOTE: this is a beta feature, the API might change.

    Args:
        service_context (ServiceContext): Service context.

    """

    service_context: ServiceContext
    pii_str_tmpl: str = DEFAULT_PII_TMPL
    pii_node_info_key: str = "__pii_node_info__"

    @classmethod
    def class_name(cls) -> str:
        return "PIINodePostprocessor"

    def mask_pii(self, text: str) -> Tuple[str, Dict]:
        """Mask PII in text."""
        pii_prompt = PromptTemplate(self.pii_str_tmpl)
        # TODO: allow customization
        task_str = (
            "Mask out the PII, replace each PII with a tag, and return the text. "
            "Return the mapping in JSON."
        )

        response = self.service_context.llm_predictor.predict(
            pii_prompt, context_str=text, query_str=task_str
        )
        splits = response.split("Output Mapping:")
        text_output = splits[0].strip()
        json_str_output = splits[1].strip()
        json_dict = json.loads(json_str_output)
        return text_output, json_dict

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        # swap out text from nodes, with the original node mappings
        new_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            new_text, mapping_info = self.mask_pii(
                node.get_content(metadata_mode=MetadataMode.LLM)
            )
            new_node = deepcopy(node)
            new_node.excluded_embed_metadata_keys.append(self.pii_node_info_key)
            new_node.excluded_llm_metadata_keys.append(self.pii_node_info_key)
            new_node.metadata[self.pii_node_info_key] = mapping_info
            new_node.set_content(new_text)
            new_nodes.append(NodeWithScore(node=new_node, score=node_with_score.score))

        return new_nodes


class NERPIINodePostprocessor(BaseNodePostprocessor):
    """NER PII Node processor.

    Uses a HF transformers model.

    """

    pii_node_info_key: str = "__pii_node_info__"

    @classmethod
    def class_name(cls) -> str:
        return "NERPIINodePostprocessor"

    def mask_pii(self, ner: Callable, text: str) -> Tuple[str, Dict]:
        """Mask PII in text."""
        new_text = text
        response = ner(text)
        mapping = {}
        for entry in response:
            entity_group_tag = f"[{entry['entity_group']}_{entry['start']}]"
            new_text = new_text.replace(entry["word"], entity_group_tag).strip()
            mapping[entity_group_tag] = entry["word"]
        return new_text, mapping

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        from transformers import pipeline

        ner = pipeline("ner", grouped_entities=True)

        # swap out text from nodes, with the original node mappings
        new_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            new_text, mapping_info = self.mask_pii(
                ner, node.get_content(metadata_mode=MetadataMode.LLM)
            )
            new_node = deepcopy(node)
            new_node.excluded_embed_metadata_keys.append(self.pii_node_info_key)
            new_node.excluded_llm_metadata_keys.append(self.pii_node_info_key)
            new_node.metadata[self.pii_node_info_key] = mapping_info
            new_node.set_content(new_text)
            new_nodes.append(NodeWithScore(node=new_node, score=node_with_score.score))

        return new_nodes
