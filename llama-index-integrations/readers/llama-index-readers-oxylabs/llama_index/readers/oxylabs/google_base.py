from dataclasses import dataclass
from typing import Optional, Any

from llama_index.core import Document

from oxylabs.sources.response import Response

from llama_index.readers.oxylabs.base import OxylabsBaseReader


RESULT_CATEGORIES = [
    "knowledge_graph",
    "combined_search_result",
    "product_information",
    "local_information",
    "search_information",
]


@dataclass
class ResponseElement:
    tag: str
    display_tag: str
    path_: str
    python_type: str
    parent: Optional["ResponseElement"]


class OxylabsGoogleBaseReader(OxylabsBaseReader):
    parsing_recursion_depth: int = 5
    image_binary_content_attributes: list[str] = ["image_data", "data"]
    excluded_result_attributes: list[str] = ["pos_overall"]
    image_binary_content_array_attribute: str = "images"
    binary_content_replacement: str = "Redacted base64 image string..."
    include_binary_image_data: bool = False

    def __init__(self, username: str, password: str, **data) -> None:
        super().__init__(username=username, password=password, **data)

    def _get_document_from_response(
        self, response: list[dict] | list[list[dict]]
    ) -> Document:
        processed_content = self._process_responses(response)
        return Document(text=processed_content)

    def get_response(self, payload: dict) -> Response:
        raise NotImplementedError(
            "Not implemented in the base class! Use one the child classes instead!"
        )

    async def aget_response(self, payload: dict) -> Response:
        raise NotImplementedError(
            "Not implemented in the base class! Use one the child classes instead!"
        )

    @staticmethod
    def validate_response_categories(result_categories: list) -> list:
        validated_categories = []
        for result_category in result_categories:
            if result_category in RESULT_CATEGORIES:
                validated_categories.append(result_category)

        return validated_categories

    def _process_responses(self, res: list[dict], **kwargs: Any) -> str:
        result_ = "No good search result found"

        result_category_processing_map = {
            "knowledge_graph": self._create_knowledge_graph_snippets,
            "combined_search_result": self._create_combined_search_result_snippets,
            "product_information": self._create_product_information_snippets,
            "local_information": self._create_local_information_snippets,
            "search_information": self._create_search_information_snippets,
        }

        snippets: list[str] = []
        validated_categories = self.validate_response_categories(
            kwargs.get("result_categories", [])
        )
        result_categories_ = validated_categories or []

        for validated_response in res:
            if result_categories_:
                for result_category in result_categories_:
                    result_category_processing_map[result_category](
                        validated_response, snippets
                    )
            else:
                for result_category in result_category_processing_map:
                    result_category_processing_map[result_category](
                        validated_response, snippets
                    )

        if snippets:
            result_ = "\n\n".join(snippets)

        return result_

    def _process_tags(
        self, snippets_: list, tags_: list, results: dict, group_name: str = ""
    ) -> None:
        check_tags = [tag_[0] in results for tag_ in tags_]
        if any(check_tags):
            for tag in tags_:
                tag_content = results.get(tag[0], {}) or {}
                if tag_content:
                    collected_snippets = self._recursive_snippet_collector(
                        tag_content,
                        max_depth=self.parsing_recursion_depth,
                        current_depth=0,
                        parent_=ResponseElement(
                            path_=f"{group_name}-{tag[0]}",
                            tag=tag[0],
                            display_tag=tag[1],
                            python_type=str(type(tag_content)),
                            parent=None,
                        ),
                    )
                    if collected_snippets:
                        snippets_.append(collected_snippets)

    def _recursive_snippet_collector(
        self,
        target_structure: Any,
        max_depth: int,
        current_depth: int,
        parent_: ResponseElement,
    ) -> str:
        target_snippets: list[str] = []

        padding_multiplier = current_depth + 1
        recursion_padding = "  " * padding_multiplier

        if current_depth >= max_depth:
            return "\n".join(target_snippets)

        if isinstance(target_structure, (str, float, int)):
            self._recursion_process_simple_types(
                parent_, recursion_padding, target_snippets, target_structure
            )

        elif isinstance(target_structure, dict):
            self.recursion_process_dict(
                current_depth,
                max_depth,
                parent_,
                recursion_padding,
                target_snippets,
                target_structure,
            )

        elif isinstance(target_structure, (list, tuple)):
            self.recursion_process_array(
                current_depth,
                max_depth,
                parent_,
                recursion_padding,
                target_snippets,
                target_structure,
            )

        return "\n".join(target_snippets)

    def recursion_process_array(
        self,
        current_depth: int,
        max_depth: int,
        parent_: ResponseElement,
        recursion_padding: str,
        target_snippets: list,
        target_structure: Any,
    ) -> None:
        if target_structure:
            target_snippets.append(
                f"{recursion_padding}{parent_.display_tag.upper()} ITEMS: "
            )
        for nr_, element_ in enumerate(target_structure):
            target_snippets.append(
                self._recursive_snippet_collector(
                    element_,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    parent_=ResponseElement(
                        path_=f"{parent_.path_.upper()}-ITEM-{nr_ + 1}",
                        tag=parent_.tag.upper(),
                        display_tag=f"{parent_.tag.upper()}-ITEM-{nr_ + 1}",
                        python_type=str(type(target_structure)),
                        parent=parent_,
                    ),
                )
            )

    def recursion_process_dict(
        self,
        current_depth: int,
        max_depth: int,
        parent_: ResponseElement,
        recursion_padding: str,
        target_snippets: list,
        target_structure: Any,
    ) -> None:
        if not target_structure:
            return

        target_snippets.append(f"{recursion_padding}{parent_.display_tag.upper()}: ")
        for key_, value_ in target_structure.items():
            if isinstance(value_, dict) and value_:
                target_snippets.append(f"{recursion_padding}{key_.upper()}: ")
                target_snippets.append(
                    self._recursive_snippet_collector(
                        value_,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        parent_=ResponseElement(
                            path_=f"{parent_.path_.upper()}-{key_.upper()}",
                            tag=key_.upper(),
                            display_tag=key_.upper(),
                            python_type=str(type(value_)),
                            parent=parent_,
                        ),
                    )
                )

            elif isinstance(value_, (list, tuple)) and value_:
                target_snippets.append(f"{recursion_padding}{key_.upper()} ITEMS: ")
                for nr_, _element in enumerate(value_):
                    target_snippets.append(
                        self._recursive_snippet_collector(
                            _element,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            parent_=ResponseElement(
                                path_=f"{parent_.path_.upper()}"
                                f"-{key_.upper()}-ITEM-{nr_ + 1}",
                                tag=key_.upper(),
                                display_tag=f"{key_.upper()}-ITEM-{nr_ + 1}",
                                python_type=str(type(value_)),
                                parent=parent_,
                            ),
                        )
                    )

            elif isinstance(value_, (str, float, int)) and value_:
                if (
                    key_ in self.image_binary_content_attributes
                    and not self.include_binary_image_data
                ):
                    value_ = self.binary_content_replacement

                if key_ not in self.excluded_result_attributes:
                    target_snippets.append(
                        f"{recursion_padding}{key_.upper()}: {value_!s}"
                    )

    def _recursion_process_simple_types(
        self,
        parent_: ResponseElement,
        recursion_padding: str,
        target_snippets: list,
        target_structure: Any,
    ) -> None:
        if not target_structure:
            return

        if parent_.python_type == str(type([])):
            if (
                self.image_binary_content_array_attribute.upper()
                in parent_.path_.split("-")[-3:]
                or parent_.tag.lower() in self.image_binary_content_attributes
            ) and not self.include_binary_image_data:
                target_structure = self.binary_content_replacement

            target_snippets.append(
                f"{recursion_padding}{parent_.display_tag}: {target_structure!s}"
            )

        elif parent_.python_type == str(type({})):
            if (
                parent_.tag.lower() in self.image_binary_content_attributes
                and not self.include_binary_image_data
            ):
                target_structure = self.binary_content_replacement

            if parent_.tag.lower() not in self.excluded_result_attributes:
                target_snippets.append(
                    f"{recursion_padding}{parent_.display_tag}: {target_structure!s}"
                )

    def _create_knowledge_graph_snippets(
        self, results: dict, knowledge_graph_snippets: list
    ) -> None:
        knowledge_graph_tags = [
            ("knowledge", "Knowledge Graph"),
            ("recipes", "Recipes"),
            ("item_carousel", "Item Carousel"),
            ("apps", "Apps"),
        ]
        self._process_tags(
            knowledge_graph_snippets, knowledge_graph_tags, results, "Knowledge"
        )

    def _create_combined_search_result_snippets(
        self, results: dict, combined_search_result_snippets: list
    ) -> None:
        combined_search_result_tags = [
            ("organic", "Organic Results"),
            ("organic_videos", "Organic Videos"),
            ("paid", "Paid Results"),
            ("featured_snipped", "Feature Snipped"),
            ("top_stories", "Top Stories"),
            ("finance", "Finance"),
            ("sports_games", "Sports Games"),
            ("twitter", "Twitter"),
            ("discussions_and_forums", "Discussions and Forums"),
            ("images", "Images"),
            ("videos", "Videos"),
            ("video_box", "Video box"),
        ]
        self._process_tags(
            combined_search_result_snippets,
            combined_search_result_tags,
            results,
            "Combined Search Results",
        )

    def _create_product_information_snippets(
        self, results: dict, product_information_snippets: list
    ) -> None:
        product_information_tags = [
            ("popular_products", "Popular Products"),
            ("pla", "Product Listing Ads (PLA)"),
        ]
        self._process_tags(
            product_information_snippets,
            product_information_tags,
            results,
            "Product Information",
        )

    def _create_local_information_snippets(
        self, results: dict, local_information_snippets: list
    ) -> None:
        local_information_tags = [
            ("top_sights", "Top Sights"),
            ("flights", "Flights"),
            ("hotels", "Hotels"),
            ("local_pack", "Local Pack"),
            ("local_service_ads", "Local Service Ads"),
            ("jobs", "Jobs"),
        ]
        self._process_tags(
            local_information_snippets,
            local_information_tags,
            results,
            "Local Information",
        )

    def _create_search_information_snippets(
        self, results: dict, search_information_snippets: list
    ) -> None:
        search_information_tags = [
            ("search_information", "Search Information"),
            ("related_searches", "Related Searches"),
            ("related_searches_categorized", "Related Searches Categorized"),
            ("related_questions", "Related Questions"),
        ]
        self._process_tags(
            search_information_snippets,
            search_information_tags,
            results,
            "Search Information",
        )
