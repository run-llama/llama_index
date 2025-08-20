import inspect
import warnings
from datetime import datetime
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from llama_index_instrumentation.span.simple import SimpleSpan

from .base import BaseSpanHandler

if TYPE_CHECKING:
    from treelib import Tree


class SimpleSpanHandler(BaseSpanHandler[SimpleSpan]):
    """Span Handler that manages SimpleSpan's."""

    def class_name(cls) -> str:
        """Class name."""
        return "SimpleSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        """Create a span."""
        return SimpleSpan(id_=id_, parent_id=parent_span_id, tags=tags or {})

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        """Logic for preparing to drop a span."""
        span = self.open_spans[id_]
        span = cast(SimpleSpan, span)
        span.end_time = datetime.now()
        span.duration = (span.end_time - span.start_time).total_seconds()
        with self.lock:
            self.completed_spans += [span]
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """Logic for droppping a span."""
        if id_ in self.open_spans:
            with self.lock:
                span = self.open_spans[id_]
                span.metadata = {"error": str(err)}
                self.dropped_spans += [span]
            return span

        return None

    def _get_parents(self) -> List[SimpleSpan]:
        """Helper method to get all parent/root spans."""
        all_spans = self.completed_spans + self.dropped_spans
        return [s for s in all_spans if s.parent_id is None]

    def _build_tree_by_parent(
        self, parent: SimpleSpan, acc: List[SimpleSpan], spans: List[SimpleSpan]
    ) -> List[SimpleSpan]:
        """Builds the tree by parent root."""
        if not spans:
            return acc

        children = [s for s in spans if s.parent_id == parent.id_]
        if not children:
            return acc
        updated_spans = [s for s in spans if s not in children]

        children_trees = [
            self._build_tree_by_parent(
                parent=c, acc=[c], spans=[s for s in updated_spans if c != s]
            )
            for c in children
        ]

        return acc + reduce(lambda x, y: x + y, children_trees)

    def _get_trace_trees(self) -> List["Tree"]:
        """Method for getting trace trees."""
        try:
            from treelib import Tree
        except ImportError as e:
            raise ImportError(
                "`treelib` package is missing. Please install it by using "
                "`pip install treelib`."
            )

        all_spans = self.completed_spans + self.dropped_spans
        for s in all_spans:
            if s.parent_id is None:
                continue
            if not any(ns.id_ == s.parent_id for ns in all_spans):
                warnings.warn(f"Parent with id {s.parent_id} missing from spans")
                s.parent_id += "-MISSING"
                all_spans.append(SimpleSpan(id_=s.parent_id, parent_id=None))

        parents = self._get_parents()
        span_groups = []
        for p in parents:
            this_span_group = self._build_tree_by_parent(
                parent=p, acc=[p], spans=[s for s in all_spans if s != p]
            )
            sorted_span_group = sorted(this_span_group, key=lambda x: x.start_time)
            span_groups.append(sorted_span_group)

        trees = []
        tree = Tree()
        for grp in span_groups:
            for span in grp:
                if span.parent_id is None:
                    # complete old tree unless its empty (i.e., start of loop)
                    if tree.all_nodes():
                        trees.append(tree)
                        # start new tree
                        tree = Tree()

                tree.create_node(
                    tag=f"{span.id_} ({span.duration})",
                    identifier=span.id_,
                    parent=span.parent_id,
                    data=span.start_time,
                )

        trees.append(tree)
        return trees

    def print_trace_trees(self) -> None:
        """Method for viewing trace trees."""
        trees = self._get_trace_trees()
        for tree in trees:
            print(tree.show(stdout=False, sorting=True, key=lambda node: node.data))
            print("")
