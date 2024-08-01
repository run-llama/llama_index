from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)
from llama_index.core.schema import TextNode
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from unittest import TestCase, SkipTest


def get_store(
    props_schema: str = "`key` STRING, `_node_content` STRING, `_node_type` STRING, `document_id` STRING, `doc_id` STRING, `ref_doc_id` STRING",
):
    return NebulaPropertyGraphStore(
        space="test_property_graph_store",
        overwrite=True,
        props_schema=props_schema,
    )


class TestPropertyGraphStore(TestCase):
    @classmethod
    def setUp(self) -> None:
        try:
            g = get_store()
        except RuntimeError:
            raise SkipTest("NebulaGraph service is not available")

    def test_add(self) -> None:
        g = get_store()

        e1 = EntityNode(name="e1")
        e2 = EntityNode(name="e2")
        r = Relation(label="r", source_id=e1.id, target_id=e2.id)

        g.upsert_nodes([e1, e2])
        g.upsert_relations([r])
        triplets = g.get_triplets(entity_names=["e1"])

        assert len(triplets) == 1

    def test_delete(self) -> None:
        g = get_store()

        e1 = EntityNode(name="e1")
        e2 = EntityNode(name="e2")
        r = Relation(label="r", source_id=e1.id, target_id=e2.id)

        g.upsert_nodes([e1, e2])
        g.upsert_relations([r])
        g.delete(ids=[e1.id])

        assert len(g.get_triplets()) == 0

    def test_get(self) -> None:
        g = get_store()

        e1 = EntityNode(name="e1")
        e2 = EntityNode(name="e2", properties={"key": "value"})
        r = Relation(label="r", source_id=e1.id, target_id=e2.id)

        g.upsert_nodes([e1, e2])
        g.upsert_relations([r])

        assert g.get_triplets() == []
        assert g.get_triplets(entity_names=["e1"]) == [(e1, r, e2)]
        assert g.get_triplets(entity_names=["e2"]) == [(e1, r, e2)]
        assert g.get_triplets(relation_names=["r"]) == [(e1, r, e2)]
        assert g.get_triplets(properties={"key": "value"}) == [(e1, r, e2)]

    def test_delete_node_by_node_ids(self) -> None:
        g = get_store()

        n1 = TextNode(id_="n1", text="n1")
        n2 = TextNode(id_="n2", text="n2")

        g.upsert_llama_nodes([n1, n2])
        g.delete_llama_nodes(node_ids=["n1"])

        all_nodes = g.get_all_nodes()
        assert len(all_nodes) == 1

    def test_delete_node_by_ref_doc_ids(self) -> None:
        # TODO: Not yet passed for now
        pass

    #     g = get_store()

    #     n1 = TextNode(
    #         id_="n1",
    #         text="n1",
    #         relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n2")},
    #     )
    #     n2 = TextNode(id_="n2", text="n2")

    #     g.upsert_llama_nodes([n1, n2])
    #     g.delete_llama_nodes(ref_doc_ids=["n2"])

    #     all_nodes = g.get_all_nodes()
    #     assert len(all_nodes) == 0

    #     # g = SimplePropertyGraphStore()

    #     n1 = TextNode(
    #         id_="n1",
    #         text="n1",
    #         relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="n3")},
    #     )
    #     n2 = TextNode(id_="n2", text="n2")

    #     g.upsert_llama_nodes([n1, n2])
    #     g.delete_llama_nodes(ref_doc_ids=["n3"])

    #     all_nodes = g.get_all_nodes()
    #     assert len(all_nodes) == 1

    def test_get_nodes(self) -> None:
        g = get_store()

        n1 = TextNode(id_="n1", text="n1")
        n2 = TextNode(id_="n2", text="n2")

        g.upsert_llama_nodes([n1, n2])
        retrieved = g.get_llama_nodes(["n1", "n2"])

        assert retrieved == [n1, n2] or retrieved == [n2, n1]
