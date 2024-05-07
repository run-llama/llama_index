from llama_index.core.schema import TextNode


class VespaNode(TextNode):
    vespa_fields: dict