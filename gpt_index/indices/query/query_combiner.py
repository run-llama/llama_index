# """Query combiner class."""

# from abc import ABC, abstractmethod
# from typing import List
# from gpt_index.response.schema import Response
# from gpt_index.indices.query.schema import QueryBundle
# from gpt_index.indices.query.base import BaseGPTIndexQuery


# class BaseQueryCombiner:
#     """Base query combiner."""

#     @abstractmethod
#     def _combine_queries(
#         self, prev_response: Response, query_bundle: QueryBundle
#     ) -> QueryBundle:
#         """Combine queries."""
#         pass

#     def run(
#         self, query_obj: BaseGPTIndexQuery, query_bundles: List[QueryBundle]
#     ) -> Response:
#         """Run query combiner."""
#         prev_response = None
#         for query_bundle in query_bundles:
#             if prev_response is not None:
#                 updated_query_bundle = self._combine_queries(
#                     prev_response, query_bundle
#                 )
#             prev_response = query_obj.query(updated_query_bundle)

#         return prev_response


# class SingleQueryCombiner(BaseQueryCombiner):
#     """Single query combiner.

#     Only runs for a single query

#     """

#     def _combine_queries(
#         self, prev_response: Response, query_bundle: QueryBundle
#     ) -> QueryBundle:
#         """Combine queries."""
#         raise NotImplementedError

#     def run(
#         self, query_obj: BaseGPTIndexQuery, query_bundles: List[QueryBundle]
#     ) -> Response:
#         """Run query combiner."""
#         if len(query_bundles) > 1:
#             raise ValueError("Only one query bundle allowed for SingleQueryCombiner")
#         return query_obj.query(query_bundles[0])
