from sphinx.application import Sphinx
from docutils import nodes
from sphinx.environment.adapters.toctree import TocTree
from sphinx.util.matching import Matcher

from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from docutils import nodes
from docutils.nodes import Element

from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging, url_re
from sphinx.util.matching import Matcher
from sphinx.util.nodes import clean_astext, process_only_nodes
from collections.abc import Iterable

from sphinx.builders import Builder
from sphinx.environment.adapters.toctree import TocTree

# Most of the code is copied from https://github.com/sphinx-doc/sphinx/blob/3b5d2afb9a20f1bfa9a6828209fc10151905cea3/sphinx/environment/adapters/toctree.py#L8
# I will highlight the changes I made

logger = logging.getLogger(__name__)


def resolve(
    tree: TocTree,
    docname: str,
    builder: "Builder",
    toctree: addnodes.toctree,
    prune: bool = True,
    maxdepth: int = 0,
    titles_only: bool = False,
    collapse: bool = False,
    includehidden: bool = False,
):
    """Resolve a *toctree* node into individual bullet lists with titles
    as items, returning None (if no containing titles are found) or
    a new node.

    If *prune* is True, the tree is pruned to *maxdepth*, or if that is 0,
    to the value of the *maxdepth* option on the *toctree* node.
    If *titles_only* is True, only toplevel document titles will be in the
    resulting tree.
    If *collapse* is True, all branches not containing docname will
    be collapsed.
    """
    if toctree.get("hidden", False) and not includehidden:
        return None
    generated_docnames: Dict[str, Tuple[str, str, str]] = (
        tree.env.domains["std"].initial_data["labels"].copy()
    )

    # For reading the following two helper function, it is useful to keep
    # in mind the node structure of a toctree (using HTML-like node names
    # for brevity):
    #
    # <ul>
    #   <li>
    #     <p><a></p>
    #     <p><a></p>
    #     ...
    #     <ul>
    #       ...
    #     </ul>
    #   </li>
    # </ul>
    #
    # The transformation is made in two passes in order to avoid
    # interactions between marking and pruning the tree (see bug #1046).

    toctree_ancestors = tree.get_toctree_ancestors(docname)
    included = Matcher(tree.env.config.include_patterns)
    excluded = Matcher(tree.env.config.exclude_patterns)
    metadata = {}

    def _toctree_add_classes(node: Element, depth: int) -> None:
        """Add 'toctree-l%d' and 'current' classes to the toctree."""
        for subnode in node.children:
            if isinstance(subnode, (addnodes.compact_paragraph, nodes.list_item)):
                # for <p> and <li>, indicate the depth level and recurse
                subnode["classes"].append("toctree-l%d" % (depth - 1))
                _toctree_add_classes(subnode, depth)
            elif isinstance(subnode, nodes.bullet_list):
                # for <ul>, just recurse
                _toctree_add_classes(subnode, depth + 1)
            elif isinstance(subnode, nodes.reference):
                # for <a>, identify which entries point to the current
                # document and therefore may not be collapsed
                internal = subnode.get("internal", False)
                if internal:
                    if subnode.children:
                        children = subnode.children
                        text = children[0].astext()
                        if (
                            "anchorname" in subnode.attributes
                            and len(subnode.attributes["anchorname"]) == 0
                        ):
                            metadata[text] = subnode["refuri"]
                    else:
                        print(subnode.__dict__)

                if subnode["refuri"] == docname:
                    if not subnode["anchorname"]:
                        # give the whole branch a 'current' class
                        # (useful for styling it differently)
                        branchnode: Element = subnode
                        while branchnode:
                            branchnode["classes"].append("current")
                            branchnode = branchnode.parent
                    # mark the list_item as "on current page"
                    if subnode.parent.parent.get("iscurrent"):
                        # but only if it's not already done
                        return
                    while subnode:
                        subnode["iscurrent"] = True
                        subnode = subnode.parent

    def _entries_from_toctree(
        toctreenode: addnodes.toctree,
        parents: List[str],
        separate: bool = False,
        subtree: bool = False,
    ) -> List[Element]:
        """Return TOC entries for a toctree node."""
        refs = [(e[0], e[1]) for e in toctreenode["entries"]]
        entries: List[Element] = []
        for title, ref in refs:
            try:
                refdoc = None
                if url_re.match(ref):
                    if title is None:
                        title = ref
                    reference = nodes.reference(
                        "",
                        "",
                        internal=False,
                        refuri=ref,
                        anchorname="",
                        *[nodes.Text(title)]
                    )
                    para = addnodes.compact_paragraph("", "", reference)
                    item = nodes.list_item("", para)
                    toc = nodes.bullet_list("", item)
                elif ref == "self":
                    # 'self' refers to the document from which this
                    # toctree originates
                    ref = toctreenode["parent"]
                    if not title:
                        title = clean_astext(self.env.titles[ref])
                    reference = nodes.reference(
                        "",
                        "",
                        internal=True,
                        refuri=ref,
                        anchorname="",
                        *[nodes.Text(title)]
                    )
                    para = addnodes.compact_paragraph("", "", reference)
                    item = nodes.list_item("", para)
                    # don't show subitems
                    toc = nodes.bullet_list("", item)
                elif ref in generated_docnames:
                    docname, _, sectionname = generated_docnames[ref]
                    if not title:
                        title = sectionname
                    reference = nodes.reference(
                        "", title, internal=True, refuri=docname, anchorname=""
                    )
                    para = addnodes.compact_paragraph("", "", reference)
                    item = nodes.list_item("", para)
                    # don't show subitems
                    toc = nodes.bullet_list("", item)
                else:
                    if ref in parents:
                        logger.warning(
                            __(
                                "circular toctree references "
                                "detected, ignoring: %s <- %s"
                            ),
                            ref,
                            " <- ".join(parents),
                            location=ref,
                            type="toc",
                            subtype="circular",
                        )
                        continue
                    refdoc = ref
                    toc = tree.env.tocs[ref].deepcopy()
                    maxdepth = tree.env.metadata[ref].get("tocdepth", 0)
                    if ref not in toctree_ancestors or (prune and maxdepth > 0):
                        tree._toctree_prune(toc, 2, maxdepth, collapse)
                    process_only_nodes(toc, builder.tags)
                    if title and toc.children and len(toc.children) == 1:
                        child = toc.children[0]
                        for refnode in child.findall(nodes.reference):
                            if refnode["refuri"] == ref and not refnode["anchorname"]:
                                refnode.children = [nodes.Text(title)]
                if not toc.children:
                    # empty toc means: no titles will show up in the toctree
                    logger.warning(
                        __(
                            "toctree contains reference to document %r that "
                            "doesn't have a title: no link will be generated"
                        ),
                        ref,
                        location=toctreenode,
                    )
            except KeyError:
                # this is raised if the included file does not exist
                if excluded(tree.env.doc2path(ref, False)):
                    message = __("toctree contains reference to excluded document %r")
                elif not included(tree.env.doc2path(ref, False)):
                    message = __(
                        "toctree contains reference to non-included document %r"
                    )
                else:
                    message = __(
                        "toctree contains reference to nonexisting document %r"
                    )

                logger.warning(message, ref, location=toctreenode)
            else:
                # if titles_only is given, only keep the main title and
                # sub-toctrees
                if titles_only:
                    # children of toc are:
                    # - list_item + compact_paragraph + (reference and subtoc)
                    # - only + subtoc
                    # - toctree
                    children = cast(Iterable[nodes.Element], toc)

                    # delete everything but the toplevel title(s)
                    # and toctrees
                    for toplevel in children:
                        # nodes with length 1 don't have any children anyway
                        if len(toplevel) > 1:
                            subtrees = list(toplevel.findall(addnodes.toctree))
                            if subtrees:
                                toplevel[1][:] = subtrees  # type: ignore
                            else:
                                toplevel.pop(1)
                # resolve all sub-toctrees
                for subtocnode in list(toc.findall(addnodes.toctree)):
                    if not (subtocnode.get("hidden", False) and not includehidden):
                        i = subtocnode.parent.index(subtocnode) + 1
                        for entry in _entries_from_toctree(
                            subtocnode, [refdoc, *parents], subtree=True
                        ):
                            subtocnode.parent.insert(i, entry)
                            i += 1
                        subtocnode.parent.remove(subtocnode)
                if separate:
                    entries.append(toc)
                else:
                    children = cast(Iterable[nodes.Element], toc)
                    entries.extend(children)
        if not subtree and not separate:
            ret = nodes.bullet_list()
            ret += entries
            return [ret]
        return entries

    maxdepth = maxdepth or toctree.get("maxdepth", -1)
    if not titles_only and toctree.get("titlesonly", False):
        titles_only = True
    if not includehidden and toctree.get("includehidden", False):
        includehidden = True

    # NOTE: previously, this was separate=True, but that leads to artificial
    # separation when two or more toctree entries form a logical unit, so
    # separating mode is no longer used -- it's kept here for history's sake
    tocentries = _entries_from_toctree(toctree, [], separate=False)
    if not tocentries:
        return None, {}

    newnode = addnodes.compact_paragraph("", "")
    caption = toctree.attributes.get("caption")
    if caption:
        caption_node = nodes.title(caption, "", *[nodes.Text(caption)])
        caption_node.line = toctree.line
        caption_node.source = toctree.source
        caption_node.rawsource = toctree["rawcaption"]
        if hasattr(toctree, "uid"):
            # move uid to caption_node to translate it
            caption_node.uid = toctree.uid  # type: ignore
            del toctree.uid  # type: ignore
        newnode += caption_node
    newnode.extend(tocentries)
    newnode["toctree"] = True

    # prune the tree to maxdepth, also set toc depth and current classes
    _toctree_add_classes(newnode, 1)
    tree._toctree_prune(newnode, 1, maxdepth if prune else 0, collapse)

    if (
        isinstance(newnode[-1], nodes.Element) and len(newnode[-1]) == 0
    ):  # No titles found
        return None, {}

    # set the target paths in the toctrees (they are not known at TOC
    # generation time)
    for refnode in newnode.findall(nodes.reference):
        if not url_re.match(refnode["refuri"]):
            refnode["refuri"] = (
                builder.get_relative_uri(docname, refnode["refuri"])
                + refnode["anchorname"]
            )
    return newnode, metadata


def get_toctree_for(
    tree: TocTree, docname: str, builder: "Builder", collapse: bool, **kwargs: Any
) -> Optional[Element]:
    """Return the global TOC nodetree."""
    doctree = tree.env.get_doctree(tree.env.config.root_doc)
    toctrees: List[Element] = []
    if "includehidden" not in kwargs:
        kwargs["includehidden"] = True
    if "maxdepth" not in kwargs or not kwargs["maxdepth"]:
        kwargs["maxdepth"] = 0  # n
    else:
        kwargs["maxdepth"] = int(kwargs["maxdepth"])
    kwargs["collapse"] = collapse

    print("kwargs")
    print(kwargs)
    metadatas = []
    for toctreenode in doctree.findall(addnodes.toctree):
        toctree, metadata = resolve(
            tree, docname, builder, toctreenode, prune=True, **kwargs
        )
        if toctree:
            toctrees.append(toctree)
        metadatas.append(metadata)

    if not toctrees:
        return None
    result = toctrees[0]
    for toctree in toctrees[1:]:
        result.extend(toctree.children)
    return result, metadatas


def resolve_sidebar(app: Sphinx, n_doctree, n_docname):
    if n_docname == "index":
        # Your code to handle the index.rst file
        toctree = TocTree(app.env)
        node, metadatas = get_toctree_for(
            toctree, n_docname, app.builder, collapse=False, maxdepth=5
        )  # maxdepth needs to be 5 to capture all relevant subdocuments
        print(node)

        print("metadata")
        print(metadatas)


def setup(app):
    app.connect("doctree-resolved", resolve_sidebar)
