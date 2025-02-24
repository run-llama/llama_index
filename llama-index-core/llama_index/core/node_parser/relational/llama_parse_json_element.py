from typing import Any, Callable, List, Optional, Dict

from llama_index.core.node_parser.relational.base_element import (
    BaseElementNodeParser,
    Element,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser.relational.utils import md_to_df


class LlamaParseJsonNodeParser(BaseElementNodeParser):
    """Llama Parse Json format element node parser.

    Splits a json format document from LlamaParse into Text Nodes and Index Nodes
    corresponding to embedded objects (e.g. tables).

    """

    @classmethod
    def class_name(cls) -> str:
        return "LlamaParseJsonNodeParser"

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = self.extract_elements(
            node.get_content(),
            table_filters=[self.filter_table],
            node_id=node.id_,
            node_metadata=node.metadata,
        )
        table_elements = self.get_table_elements(elements)
        # extract summaries over table elements
        self.extract_table_summaries(table_elements)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        return self.get_nodes_from_elements(
            elements, node, ref_doc_text=node.get_content()
        )

    async def aget_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = self.extract_elements(
            node.get_content(),
            table_filters=[self.filter_table],
            node_id=node.id_,
            node_metadata=node.metadata,
        )
        table_elements = self.get_table_elements(elements)
        # extract summaries over table elements
        await self.aextract_table_summaries(table_elements)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        return self.get_nodes_from_elements(
            elements, node, ref_doc_text=node.get_content()
        )

    def extract_elements(
        self,
        text: str,
        mode: Optional[str] = "json",
        node_id: Optional[str] = None,
        node_metadata: Optional[Dict[str, Any]] = None,
        table_filters: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> List[Element]:
        # get node id for each node so that we can avoid using the same id for different nodes
        """Extract elements from json based nodes.

        Args:
            text: node's text content
            mode: different modes for returning different types of elements based on the selected mode
            node_id: unique id for the node
            node_metadata: metadata for the node. the json output for the nodes contains a lot of fields for elements

        """
        elements: List[Element] = []
        currentElement = None
        page_number = node_metadata.get("page") if node_metadata is not None else 0

        if mode == "json" and node_metadata is not None:
            json_items = node_metadata.get("items") or []
            for element_idx, json_item in enumerate(json_items):
                ele_type = json_item.get("type")
                if ele_type == "heading":
                    elements.append(
                        Element(
                            id=f"id_page_{page_number}_heading_{element_idx}",
                            type="heading",
                            title_level=json_item.get("lvl"),
                            element=json_item.get("value"),
                            markdown=json_item.get("md"),
                            page_number=page_number,
                        )
                    )
                elif ele_type == "text":
                    elements.append(
                        Element(
                            id=f"id_page_{page_number}_text_{element_idx}",
                            type="text",
                            element=json_item.get("value"),
                            markdown=json_item.get("md"),
                            page_number=page_number,
                        )
                    )
                elif ele_type == "table":
                    elements.append(
                        Element(
                            id=f"id_page_{page_number}_table_{element_idx}",
                            type="table",
                            element=json_item.get("rows"),
                            markdown=json_item.get("md"),
                            page_number=page_number,
                        )
                    )
        elif mode == "images" and node_metadata is not None:
            # only get images from json metadata
            images = node_metadata.get("images") or []
            for idx, image in enumerate(images):
                elements.append(
                    Element(
                        id=f"id_page_{page_number}_image_{idx}",
                        type="image",
                        element=image,
                    )
                )
        else:
            lines = text.split("\n")
            # Then parse the lines from raw text of json
            for line in lines:
                if line.startswith("```"):
                    # check if this is the end of a code block
                    if currentElement is not None and currentElement.type == "code":
                        elements.append(currentElement)
                        currentElement = None
                        # if there is some text after the ``` create a text element with it
                        if len(line) > 3:
                            elements.append(
                                Element(
                                    id=f"id_{len(elements)}",
                                    type="text",
                                    element=line.lstrip("```"),
                                )
                            )

                    elif line.count("```") == 2 and line[-3] != "`":
                        # check if inline code block (aka have a second ``` in line but not at the end)
                        if currentElement is not None:
                            elements.append(currentElement)
                        currentElement = Element(
                            id=f"id_{len(elements)}",
                            type="code",
                            element=line.lstrip("```"),
                        )
                    elif currentElement is not None and currentElement.type == "text":
                        currentElement.element += "\n" + line
                    else:
                        if currentElement is not None:
                            elements.append(currentElement)
                        currentElement = Element(
                            id=f"id_{len(elements)}", type="text", element=line
                        )

                elif currentElement is not None and currentElement.type == "code":
                    currentElement.element += "\n" + line

                elif line.startswith("|"):
                    if currentElement is not None and currentElement.type != "table":
                        if currentElement is not None:
                            elements.append(currentElement)
                        currentElement = Element(
                            id=f"id_{len(elements)}", type="table", element=line
                        )
                    elif currentElement is not None:
                        currentElement.element += "\n" + line
                    else:
                        currentElement = Element(
                            id=f"id_{len(elements)}", type="table", element=line
                        )
                elif line.startswith("#"):
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}",
                        type="title",
                        element=line.lstrip("#"),
                        title_level=len(line) - len(line.lstrip("#")),
                    )
                else:
                    if currentElement is not None and currentElement.type != "text":
                        elements.append(currentElement)
                        currentElement = Element(
                            id=f"id_{len(elements)}", type="text", element=line
                        )
                    elif currentElement is not None:
                        currentElement.element += "\n" + line
                    else:
                        currentElement = Element(
                            id=f"id_{len(elements)}", type="text", element=line
                        )
        if currentElement is not None:
            elements.append(currentElement)

        for idx, element in enumerate(elements):
            if element.type == "table":
                assert element.markdown is not None

                should_keep = True
                perfect_table = True

                # verify that the table (markdown) have the same number of columns on each rows
                table_lines = element.markdown.split("\n")
                table_columns = [len(line.split("|")) for line in table_lines]
                if len(set(table_columns)) > 1:
                    # if the table have different number of columns on each rows, it's not a perfect table
                    # we will store the raw text for such tables instead of converting them to a dataframe
                    perfect_table = False

                # verify that the table (markdown) have at least 2 rows
                if len(table_lines) < 2:
                    should_keep = False

                # apply the table filter, now only filter empty tables
                if should_keep and perfect_table and table_filters is not None:
                    should_keep = all(tf(element) for tf in table_filters)

                # if the element is a table, convert it to a dataframe
                if should_keep:
                    if perfect_table:
                        assert element.markdown is not None
                        table = md_to_df(element.markdown)

                        elements[idx] = Element(
                            id=(
                                f"id_page_{page_number}_{node_id}_{idx}"
                                if node_id
                                else f"id_{idx}"
                            ),
                            type="table",
                            element=element,
                            table=table,
                        )
                    else:
                        # for non-perfect tables, we will store the raw text
                        # and give it a different type to differentiate it from perfect tables
                        elements[idx] = Element(
                            id=(
                                f"id_page_{page_number}_{node_id}_{idx}"
                                if node_id
                                else f"id_{idx}"
                            ),
                            type="table_text",
                            element=element.element,
                            # table=table
                        )
                else:
                    elements[idx] = Element(
                        id=(
                            f"id_page_{page_number}_{node_id}_{idx}"
                            if node_id
                            else f"id_page_{page_number}_{idx}"
                        ),
                        type="text",
                        element=element.element,
                    )
            else:
                # if the element is not a table, keep it as to text
                elements[idx] = Element(
                    id=(
                        f"id_page_{page_number}_{node_id}_{idx}"
                        if node_id
                        else f"id_page_{page_number}_{idx}"
                    ),
                    type="text",
                    element=element.element,
                )

        # merge consecutive text elements together for now
        merged_elements: List[Element] = []
        for element in elements:
            if (
                len(merged_elements) > 0
                and element.type == "text"
                and merged_elements[-1].type == "text"
            ):
                if isinstance(element.element, list):
                    merged_elements[-1].element += "\n" + " ".join(
                        str(e) for e in element.element
                    )
                else:
                    merged_elements[-1].element += "\n" + element.element
            else:
                merged_elements.append(element)
        elements = merged_elements
        return merged_elements

    def filter_table(self, table_element: Any) -> bool:
        """Filter tables."""
        # convert markdown of the table to df
        table_df = md_to_df(table_element.markdown)

        # check if table_df is not None, has more than one row, and more than one column
        return table_df is not None and not table_df.empty and len(table_df.columns) > 1
