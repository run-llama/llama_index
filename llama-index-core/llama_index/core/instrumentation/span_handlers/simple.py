from typing import Any, cast, List, Optional, TYPE_CHECKING
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.span.simple import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from datetime import datetime
import warnings

if TYPE_CHECKING:
    from treelib import Tree


class SimpleSpanHandler(BaseSpanHandler[SimpleSpan]):
    """Span Handler that managest SimpleSpan's."""

    completed_spans: List[SimpleSpan] = Field(
        default_factory=list, description="List of completed spans."
    )

    def class_name(cls) -> str:
        """Class name."""
        return "SimpleSpanHandler"

    def new_span(
        self, *args, id: str, parent_span_id: Optional[str], **kwargs
    ) -> SimpleSpan:
        """Create a span."""
        return SimpleSpan(id_=id, parent_id=parent_span_id)

    def prepare_to_exit_span(
        self, *args, id: str, result: Optional[Any] = None, **kwargs
    ) -> None:
        """Logic for preparing to drop a span."""
        span = self.open_spans[id]
        span = cast(SimpleSpan, span)
        span.end_time = datetime.now()
        span.duration = (span.end_time - span.start_time).total_seconds()
        self.completed_spans += [span]

    def prepare_to_drop_span(
        self, *args, id: str, err: Optional[Exception], **kwargs
    ) -> None:
        """Logic for droppping a span."""
        if err:
            raise err

    def _get_trace_trees(self) -> List["Tree"]:
        """Method for getting trace trees."""
        try:
            from treelib import Tree
            from treelib.exceptions import NodeIDAbsentError
        except ImportError as e:
            raise ImportError(
                "`treelib` package is missing. Please install it by using "
                "`pip install treelib`."
            )
        sorted_spans = sorted(self.completed_spans, key=lambda x: x.start_time)

        trees = []
        tree = Tree()
        for span in sorted_spans:
            if span.parent_id is None:
                # complete old tree unless its empty (i.e., start of loop)
                if tree.all_nodes():
                    trees.append(tree)
                    # start new tree
                    tree = Tree()

            try:
                tree.create_node(
                    tag=f"{span.id_} ({span.duration})",
                    identifier=span.id_,
                    parent=span.parent_id,
                    data=span.start_time,
                )
            except NodeIDAbsentError:
                warnings.warn("Parent with id {span.parent_id} missing from spans")
                # create new tree and fake parent node
                trees.append(tree)
                tree = Tree()
                tree.create_node(
                    tag=f"{span.parent_id} (MISSING)",
                    identifier=span.parent_id,
                    parent=None,
                    data=span.start_time,
                )
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
