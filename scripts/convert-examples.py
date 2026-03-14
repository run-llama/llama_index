#!/usr/bin/env python3
"""
Incremental notebook-to-markdown converter for LlamaIndex examples.

Converts Jupyter notebooks from docs/examples/ to markdown with Astro-compatible
frontmatter. Uses git diff to detect changed notebooks when a --since SHA is
provided, so only new/modified notebooks are reconverted.

Integration directories (llm, embeddings, vector_stores, retrievers) are routed
to a separate --integrations-dest under framework/integrations/ instead of
examples/.

Usage:
    # Full conversion (first run or manual dispatch):
    python scripts/convert-examples.py \
        --source docs/examples \
        --dest  /path/to/developer-hub/src/content/docs/python/examples \
        --integrations-dest /path/to/developer-hub/src/content/docs/python/framework/integrations \
        --static-source docs/src/content/docs/framework/_static

    # Incremental conversion (CI — only convert what changed since a commit):
    python scripts/convert-examples.py \
        --source docs/examples \
        --dest  ... \
        --integrations-dest ... \
        --static-source ... \
        --since abc1234
"""

import argparse
import base64
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set, Tuple

from nbconvert import MarkdownExporter
import nbformat
from tqdm import tqdm

INTEGRATION_DIRS = frozenset(["llm", "embeddings", "vector_stores", "retrievers"])


# ---------------------------------------------------------------------------
# Git-based change detection
# ---------------------------------------------------------------------------


def get_changed_notebooks(source: Path, since_sha: str) -> Set[str]:
    """
    Use git diff to find notebooks changed since a given commit SHA.

    Returns relative paths (to source) of changed/added .ipynb files.
    """
    # source is e.g. /repo/docs/examples — we need the git-root-relative path
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel_source = source.relative_to(repo_root)

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACM",
            f"{since_sha}..HEAD",
            "--",
            f"{rel_source}/**/*.ipynb",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    changed = set()
    for line in result.stdout.strip().splitlines():
        if line:
            # Convert from repo-relative to source-relative
            changed.add(str(Path(line).relative_to(rel_source)))
    return changed


def get_deleted_notebooks(source: Path, since_sha: str) -> Set[str]:
    """Use git diff to find notebooks deleted since a given commit SHA."""
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel_source = source.relative_to(repo_root)

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=D",
            f"{since_sha}..HEAD",
            "--",
            f"{rel_source}/**/*.ipynb",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    deleted = set()
    for line in result.stdout.strip().splitlines():
        if line:
            deleted.add(str(Path(line).relative_to(rel_source)))
    return deleted


def find_orphaned_outputs(source: Path, dest: Path, int_dest: Path) -> List[Path]:
    """
    Find .md files in dest/int_dest that have no corresponding .ipynb in source.

    Catches deletions that happened before the --since window, or when running
    without --since.
    """
    all_source_notebooks = {
        str(nb.relative_to(source)) for nb in source.rglob("*.ipynb")
    }

    orphans = []
    for root, is_int in [(dest, False), (int_dest, True)]:
        if not root.exists():
            continue
        for md_file in root.rglob("*.md"):
            rel = md_file.relative_to(root)
            # Skip index files and non-notebook-produced markdown
            if rel.name == "index.md":
                continue
            # Check if there's a corresponding .ipynb in source
            if is_int:
                nb_rel = str(rel.with_suffix(".ipynb"))
            else:
                nb_rel = str(rel.with_suffix(".ipynb"))
            if nb_rel not in all_source_notebooks:
                orphans.append(md_file)
    return orphans


# ---------------------------------------------------------------------------
# Path routing
# ---------------------------------------------------------------------------


def is_integration(rel: str) -> bool:
    return Path(rel).parts[0] in INTEGRATION_DIRS


def output_for(rel: str, dest: Path, int_dest: Path) -> Path:
    md = Path(rel).with_suffix(".md")
    return int_dest / md if is_integration(rel) else dest / md


# ---------------------------------------------------------------------------
# Notebook -> Markdown conversion
# ---------------------------------------------------------------------------


def convert_header_to_frontmatter(md: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    if not m:
        return md
    title = m.group(1)
    return f"---\ntitle: >\n  {title}\n---\n\n" + md[m.end() :].lstrip()


def fix_static_paths(md: str) -> str:
    return re.sub(r"\.\./\.\./\.\./_static/", "../_static/", md)


def save_embedded_images(resources: dict, dest_dir: Path) -> None:
    if not resources or "outputs" not in resources:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, data in resources["outputs"].items():
        if isinstance(data, bytes):
            (dest_dir / name).write_bytes(data)
        elif isinstance(data, str) and data.startswith("data:image"):
            m = re.match(r"data:image/(\w+);base64,(.*)", data)
            if m:
                (dest_dir / f"{name}.{m.group(1)}").write_bytes(
                    base64.b64decode(m.group(2))
                )


def convert_single_notebook(src: Path, out: Path) -> Tuple[str, bool, str]:
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        nb = nbformat.read(str(src), as_version=4)
        body, resources = MarkdownExporter().from_notebook_node(nb)
        body = convert_header_to_frontmatter(body)
        body = fix_static_paths(body)
        out.write_text(body, encoding="utf-8")
        save_embedded_images(resources, out.parent)
        return str(src), True, ""
    except Exception as exc:
        return str(src), False, str(exc)


# ---------------------------------------------------------------------------
# Directory labels
# ---------------------------------------------------------------------------


def transform_dir_name(name: str) -> str:
    label = name.replace("_", " ")
    label = label[0].upper() + label[1:].lower()
    return label.replace("llm", "LLM")


def ensure_meta_files(root: Path) -> None:
    """Create _meta.yml for every sub-directory that doesn't already have one."""
    if not root.exists():
        return
    for d in sorted(root.rglob("*")):
        if not d.is_dir() or d.name in ("_static", "data"):
            continue
        meta = d / "_meta.yml"
        if not meta.exists():
            meta.write_text(f"label: {transform_dir_name(d.name)}\ncollapsed: true\n")


# ---------------------------------------------------------------------------
# Asset helpers
# ---------------------------------------------------------------------------


def copy_static(static_source: Path, dest: Path) -> None:
    target = dest / "_static"
    if static_source.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(static_source, target)
        print(f"  _static -> {target}")


def copy_data(source: Path, dest: Path) -> None:
    src = source / "data"
    dst = dest / "data"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  data -> {dst}")


def process_md(src: Path, dst: Path) -> None:
    content = src.read_text(encoding="utf-8")
    content = convert_header_to_frontmatter(content)
    content = fix_static_paths(content)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")


def sync_markdown_files(source: Path, dest: Path, int_dest: Path) -> None:
    """Copy standalone .md files (not produced from notebooks)."""
    for md_file in source.rglob("*.md"):
        rel = md_file.relative_to(source)
        # Skip data directory markdown
        if rel.parts[0] == "data":
            continue
        if is_integration(str(rel)):
            target = int_dest / rel
        else:
            target = dest / rel
        process_md(md_file, target)


def copy_directory_images(source: Path, dest: Path, int_dest: Path) -> None:
    """Copy image files that live alongside notebooks in source dirs."""
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
        for img in source.rglob(ext):
            rel = img.relative_to(source)
            if rel.parts[0] in ("data", "_static"):
                continue
            target = (int_dest if is_integration(str(rel)) else dest) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental notebook converter")
    parser.add_argument("--source", required=True, help="docs/examples directory")
    parser.add_argument("--dest", required=True, help="Examples output directory")
    parser.add_argument(
        "--integrations-dest",
        required=True,
        help="Integrations output directory (framework/integrations)",
    )
    parser.add_argument(
        "--static-source",
        required=True,
        help="Path to framework/_static directory",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Git commit SHA — only convert notebooks changed since this commit",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    int_dest = Path(args.integrations_dest).resolve()
    static_src = Path(args.static_source).resolve()

    # --- 1. Determine which notebooks to convert ---
    all_notebooks = sorted(source.rglob("*.ipynb"))

    if args.since:
        changed = get_changed_notebooks(source, args.since)
        deleted = get_deleted_notebooks(source, args.since)
        to_convert = [
            str(nb.relative_to(source))
            for nb in all_notebooks
            if str(nb.relative_to(source)) in changed
        ]
        print(
            f"Incremental mode (since {args.since[:8]}): "
            f"{len(all_notebooks)} total, {len(to_convert)} changed, "
            f"{len(deleted)} deleted"
        )
    else:
        to_convert = [str(nb.relative_to(source)) for nb in all_notebooks]
        deleted = set()
        print(f"Full conversion: {len(to_convert)} notebooks")

    # --- 2. Convert changed / new notebooks ---
    if to_convert:
        pairs: List[Tuple[Path, Path]] = [
            (source / r, output_for(r, dest, int_dest)) for r in to_convert
        ]
        errors: List[Tuple[str, str]] = []
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(convert_single_notebook, s, o): r
                for (s, o), r in zip(pairs, to_convert)
            }
            with tqdm(total=len(futures), desc="Converting") as pbar:
                for fut in as_completed(futures):
                    path, ok, err = fut.result()
                    pbar.update(1)
                    if not ok:
                        errors.append((path, err))
                        print(f"\n  ERROR {path}: {err}")
        if errors:
            print(f"\n{len(errors)} notebook(s) failed to convert.")

    # --- 3. Handle deletions ---
    # Delete outputs for notebooks removed in the git diff
    for rel in deleted:
        p = output_for(rel, dest, int_dest)
        if p.exists():
            p.unlink()
            print(f"  Deleted {p}")

    # Also clean orphaned outputs (catches older deletions or first-run mismatches)
    orphans = find_orphaned_outputs(source, dest, int_dest)
    for orphan in orphans:
        orphan.unlink()
        print(f"  Cleaned orphan {orphan}")

    # Clean empty directories left behind
    for root in (dest, int_dest):
        if not root.exists():
            continue
        for d in sorted(root.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    # --- 4. Copy assets (always — fast) ---
    print("Syncing assets...")
    dest.mkdir(parents=True, exist_ok=True)
    int_dest.mkdir(parents=True, exist_ok=True)
    copy_static(static_src, dest)
    copy_data(source, dest)
    copy_directory_images(source, dest, int_dest)
    sync_markdown_files(source, dest, int_dest)

    # --- 5. _meta.yml ---
    ensure_meta_files(dest)
    ensure_meta_files(int_dest)
    # Integrations root needs a specific sort order
    (int_dest / "_meta.yml").write_text(
        "label: Integrations\ncollapsed: true\norder: 998\n"
    )

    print("Done.")


if __name__ == "__main__":
    main()
