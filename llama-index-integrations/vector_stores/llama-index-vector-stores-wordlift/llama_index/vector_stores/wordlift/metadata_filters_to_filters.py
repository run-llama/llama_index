from llama_index.core.vector_stores import (
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from wordlift_client import Filter, FilterValue


class MetadataFiltersToFilters:
    @staticmethod
    def metadata_filters_to_filters(metadata_filters: MetadataFilters):
        # Return an empty list if there are no filters.
        if (
            not hasattr(metadata_filters, "filters")
            or len(metadata_filters.filters) == 0
        ):
            return []

        # Only one filter.
        if len(metadata_filters.filters) == 1:
            metadata_filter = metadata_filters.filters[0]
            return [
                Filter(
                    key=metadata_filter.key,
                    operator=MetadataFiltersToFilters.metadata_filter_operator_to_filter_operator(
                        metadata_filter.operator
                    ),
                    value=FilterValue(metadata_filter.value),
                )
            ]

        # Prepare the list of filters.
        filters = []
        for metadata_filter in metadata_filters.filters:
            filters.append(
                Filter(
                    key=metadata_filter.key,
                    operator=MetadataFiltersToFilters.metadata_filter_operator_to_filter_operator(
                        metadata_filter.operator
                    ),
                    value=FilterValue(metadata_filter.value),
                )
            )

        # Join the filters abed on the metadata filter condition.
        return [
            Filter(
                operator=MetadataFiltersToFilters.metadata_filter_condition_to_filter_operators(
                    metadata_filters.condition
                ),
                filters=filters,
            )
        ]

    @staticmethod
    def metadata_filter_operator_to_filter_operator(filter_operator: FilterOperator):
        # 'EQ', 'GT', 'LT', 'NE', 'GTE', 'LTE', 'IN', 'NIN', 'AND', 'OR'
        if filter_operator == FilterOperator.EQ:
            return "EQ"  # default operator (string, int, float)
        elif filter_operator == FilterOperator.GT:
            return "GT"  # greater than (int, float)
        elif filter_operator == FilterOperator.LT:
            return "LT"  # less than (int, float)
        elif filter_operator == FilterOperator.NE:
            return "NE"  # not equal to (string, int, float)
        elif filter_operator == FilterOperator.GTE:
            return "GTE"  # greater than or equal to (int, float)
        elif filter_operator == FilterOperator.LTE:
            return "LTE"  # less than or equal to (int, float)
        elif filter_operator == FilterOperator.IN:
            return "IN"  # In array (string or number)
        elif filter_operator == FilterOperator.NIN:
            return "NIN"  # Not in array (string or number)
        else:
            raise ValueError(f"Invalid filter operator: {filter_operator}")

    @staticmethod
    def metadata_filter_condition_to_filter_operators(
        filter_condition: FilterCondition,
    ):
        if filter_condition == FilterCondition.AND:
            return "AND"
        elif filter_condition == FilterCondition.OR:
            return "OR"
        else:
            raise ValueError(f"Invalid filter condition: {filter_condition}")
