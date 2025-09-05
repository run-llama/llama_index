#!/usr/bin/env python3
"""
Migration script to find broken links in markdown files.

This script walks through all markdown files in /docs/docs and identifies broken links
based on the following criteria:
- Non-absolute URLs are considered broken
- Absolute URLs that don't start with /python/framework, /python/examples, or /python/workflows are broken
- External links (http/https, mailto) and localhost links are considered fine
- /python/workflows/ links are treated as valid (they exist in a different repository)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict
from urllib.parse import urlparse


def extract_markdown_links(content: str) -> List[Tuple[str, int, str]]:
    """
    Extract all markdown links from content.
    Returns list of (clean_url, line_number, original_url) tuples.
    """
    links = []
    lines = content.split('\n')
    
    # Pattern for markdown links: [text](url)
    link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
    
    for line_num, line in enumerate(lines, 1):
        matches = link_pattern.findall(line)
        for text, url in matches:
            # Clean up the URL (remove fragments and query params for analysis)
            clean_url = url.split('#')[0].split('?')[0].strip()
            if clean_url:  # Only include non-empty URLs
                links.append((clean_url, line_num, url))
    
    return links


def is_external_link(url: str) -> bool:
    """Check if a URL is an external link (http/https, mailto, or localhost)."""
    parsed = urlparse(url)
    return (
        parsed.scheme in ('http', 'https', 'mailto') or 
        'localhost' in url or 
        parsed.netloc != ''
    )


def normalize_url_to_file_path(url: str, docs_dir: Path) -> Optional[Path]:
    """
    Convert a URL to the expected file path based on the rules:
    - URLs should be lowercase
    - /path/blah should map to /path/blah.md
    - /path should map to /path/index.md  
    - /path/blah should also check for /path/blah.ipynb
    
    Returns the Path object if a matching file exists, None otherwise.
    """
    if not url.startswith('/python/'):
        return None
    
    # Handle /python/workflows/ URLs - they exist in a different repo, so treat as valid
    if url.startswith('/python/workflows/'):
        # Return a dummy path to indicate this is valid but external
        return Path('__workflows_external__')
    
    # Remove /python/framework or /python/examples prefix to get the docs-relative path
    if url.startswith('/python/framework'):
        relative_path = url[len('/python/framework'):]
        # Framework URLs map directly to docs root
        base_path = docs_dir
    elif url.startswith('/python/examples'):
        relative_path = url[len('/python/examples'):]
        # Examples URLs map to docs/examples/
        base_path = docs_dir / 'examples'
    else:
        return None
    
    # Remove leading slash if present
    if relative_path.startswith('/'):
        relative_path = relative_path[1:]
    
    # If empty path, it should point to index.md at the root
    if not relative_path:
        relative_path = 'index'
    
    # Convert to lowercase as per requirements
    relative_path = relative_path.lower()
    
    # Try different file extensions and locations
    possible_files = []
    
    if relative_path.endswith('/'):
        # Path ends with slash, should map to index.md in that directory
        relative_path = relative_path.rstrip('/')
        possible_files.extend([
            base_path / f"{relative_path}/index.md",
            base_path / f"{relative_path}.md"
        ])
    else:
        # Check if this already has a file extension (like .png, .jpg, etc.)
        if '.' in relative_path and relative_path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'svg', 'pdf', 'txt', 'json', 'csv', 'html', 'css', 'js']:
            # This is a static asset, check if it exists as-is
            possible_files.append(base_path / relative_path)
        else:
            # Try as direct file with extensions
            possible_files.extend([
                base_path / f"{relative_path}.md",
                base_path / f"{relative_path}.ipynb",
                base_path / f"{relative_path}/index.md"
            ])
    
    # Check if any of the possible files exist
    for file_path in possible_files:
        if file_path.exists() and file_path.is_file():
            return file_path
    
    return None


def is_broken_link(url: str, docs_dir: Optional[Path] = None) -> bool:
    """
    Determine if a link is broken based on the migration criteria:
    - Non-absolute URLs are broken
    - Absolute URLs that don't start with /python/framework or /python/examples are broken
    - External links are fine
    - For valid absolute URLs, check if the target file actually exists
    """
    # External links are fine
    if is_external_link(url):
        return False
    
    # Empty or just fragment links
    if not url or url.startswith('#'):
        return False
    
    # Non-absolute URLs (relative paths) are broken
    if not url.startswith('/'):
        return True
    
    # Absolute URLs that don't start with allowed prefixes are broken
    allowed_prefixes = ['/python/framework', '/python/examples', '/python/workflows']
    if not any(url.startswith(prefix) for prefix in allowed_prefixes):
        return True
    
    # If we have docs_dir, check if the file actually exists
    if docs_dir is not None:
        target_file = normalize_url_to_file_path(url, docs_dir)
        if target_file is None:
            return True  # File doesn't exist
    
    return False


def find_markdown_files(docs_dir: str) -> List[str]:
    """Find all markdown files in the docs directory."""
    markdown_files = []
    docs_path = Path(docs_dir)
    
    for file_path in docs_path.rglob('*.md'):
        markdown_files.append(str(file_path))
    
    return sorted(markdown_files)


def check_file_for_broken_links(file_path: str, docs_dir: Path) -> List[Tuple[str, int, str]]:
    """Check a single markdown file for broken links."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []
    
    links = extract_markdown_links(content)
    broken_links = []
    
    for clean_url, line_num, original_url in links:
        # Check if URL needs lowercasing (for absolute URLs that aren't already lowercase)
        needs_lowercasing = False
        if clean_url.startswith('/python/') and clean_url != clean_url.lower():
            needs_lowercasing = True
        
        if is_broken_link(clean_url, docs_dir) or needs_lowercasing:
            # Determine the reason why it's broken
            reason = "unknown"
            if needs_lowercasing and not is_broken_link(clean_url.lower(), docs_dir):
                reason = "needs_lowercase"
            elif is_external_link(clean_url):
                reason = "external"  # This shouldn't happen since external links are not broken
            elif not clean_url or clean_url.startswith('#'):
                reason = "fragment"  # This shouldn't happen either
            elif not clean_url.startswith('/'):
                reason = "relative_path"
            elif not any(clean_url.startswith(prefix) for prefix in ['/python/framework', '/python/examples', '/python/workflows']):
                reason = "wrong_prefix"
            else:
                # Must be a file existence issue
                target_file = normalize_url_to_file_path(clean_url, docs_dir)
                if target_file is None:
                    reason = "file_not_found"
                else:
                    reason = "unknown"
            
            broken_links.append((original_url, line_num, reason))
    
    return broken_links


def convert_relative_to_absolute(url: str, current_file_path: Path, docs_dir: Path) -> Optional[str]:
    """
    Convert a relative URL to an absolute URL if possible.
    
    Args:
        url: The relative URL to convert
        current_file_path: Path to the file containing the link
        docs_dir: Path to the docs directory
    
    Returns:
        The converted absolute URL, or None if conversion not possible
    """
    if not url or url.startswith('/') or is_external_link(url) or url.startswith('#'):
        return None
    
    # Get the directory containing the current file, relative to docs_dir
    try:
        current_file_rel = current_file_path.relative_to(docs_dir)
        current_dir = current_file_rel.parent
    except ValueError:
        return None
    
    # Resolve the relative path from the current file's directory
    try:
        # Create a Path object from the relative URL
        relative_path = Path(url)
        
        # Resolve it relative to the current file's directory (but keep it relative to docs)
        resolved_path = current_dir / relative_path
        
        # Normalize the path without making it absolute
        parts = []
        for part in resolved_path.parts:
            if part == '..':
                if parts:
                    parts.pop()
            elif part != '.':
                parts.append(part)
        
        if not parts:
            return None
            
        resolved_str = '/'.join(parts)
        
        # Check if this resolves to something in the examples directory
        if resolved_str.startswith('examples/'):
            # Extract the part after 'examples/'
            examples_part = resolved_str[len('examples/'):]
            
            # Only remove file extension if it's .md or .ipynb for the URL
            # Keep extensions for static assets
            if examples_part.endswith('.md'):
                examples_part = examples_part[:-3]
            elif examples_part.endswith('.ipynb'):
                examples_part = examples_part[:-6]
            
            return f"/python/examples/{examples_part}"
        
        # Check if this resolves to workflows (different repo)
        elif resolved_str.startswith('workflows/'):
            # Extract the part after 'workflows/'
            workflows_part = resolved_str[len('workflows/'):]
            
            # Only remove file extension if it's .md or .ipynb for the URL
            # Keep extensions for static assets
            if workflows_part.endswith('.md'):
                workflows_part = workflows_part[:-3]
            elif workflows_part.endswith('.ipynb'):
                workflows_part = workflows_part[:-6]
            
            return f"/python/workflows/{workflows_part}"
        
        # Check if this resolves to something that should be in the framework
        else:
            # Only remove file extension if it's .md or .ipynb for the URL
            # Keep extensions for static assets
            if resolved_str.endswith('.md'):
                resolved_str = resolved_str[:-3]
            elif resolved_str.endswith('.ipynb'):
                resolved_str = resolved_str[:-6]
            
            return f"/python/framework/{resolved_str}"
            
    except Exception:
        return None


def fix_links_in_file(file_path: str, docs_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Fix broken links in a single markdown file.
    
    Args:
        file_path: Path to the markdown file
        docs_dir: Path to the docs directory
        dry_run: If True, don't actually modify the file
    
    Returns:
        Dictionary with counts of fixes by type
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return {}
    
    content = original_content
    current_file_path = Path(file_path)
    
    # Extract all links with their line numbers
    links = extract_markdown_links(content)
    
    fixes = {
        "examples_converted": 0,
        "framework_converted": 0,
        "workflows_converted": 0,
        "lowercased": 0,
        "failed_conversions": 0
    }
    
    # Process links in reverse order to maintain line numbers
    lines = content.split('\n')
    
    for clean_url, line_num, original_url in reversed(links):
        # Check if this is an absolute URL that just needs lowercasing
        needs_lowercasing = (clean_url.startswith('/python/') and 
                           clean_url != clean_url.lower() and 
                           not is_broken_link(clean_url.lower(), docs_dir))
        
        if is_broken_link(clean_url, docs_dir) or needs_lowercasing:
            if needs_lowercasing:
                # Just lowercase the URL
                new_clean_url = clean_url.lower()
            else:
                # Try to convert relative URL to absolute
                new_clean_url = convert_relative_to_absolute(clean_url, current_file_path, docs_dir)
            
            if new_clean_url:
                # Verify the new URL is not broken (unless it's just lowercasing)
                if needs_lowercasing or not is_broken_link(new_clean_url, docs_dir):
                    # Preserve any fragment from the original URL
                    if '#' in original_url:
                        fragment = original_url.split('#', 1)[1]
                        new_url = f"{new_clean_url}#{fragment}"
                    else:
                        new_url = new_clean_url
                    
                    # Replace the URL in the specific line
                    line_idx = line_num - 1  # Convert to 0-based index
                    if line_idx < len(lines):
                        # Use regex to replace the specific URL in the markdown link
                        line = lines[line_idx]
                        # Pattern to match [text](original_url) exactly
                        escaped_original_url = re.escape(original_url)
                        pattern = r'\[([^\]]*)\]\(' + escaped_original_url + r'\)'
                        replacement = r'[\1](' + new_url + ')'
                        new_line = re.sub(pattern, replacement, line)
                        
                        if new_line != line:
                            lines[line_idx] = new_line
                            if needs_lowercasing:
                                fixes["lowercased"] += 1
                            elif new_url.startswith('/python/examples/'):
                                fixes["examples_converted"] += 1
                            elif new_url.startswith('/python/framework/'):
                                fixes["framework_converted"] += 1
                            elif new_url.startswith('/python/workflows/'):
                                fixes["workflows_converted"] += 1
                        else:
                            fixes["failed_conversions"] += 1
                else:
                    fixes["failed_conversions"] += 1
            else:
                fixes["failed_conversions"] += 1
    
    # Write the modified content back to the file
    if not dry_run and (fixes["examples_converted"] > 0 or fixes["framework_converted"] > 0 or fixes["workflows_converted"] > 0 or fixes["lowercased"] > 0):
        new_content = '\n'.join(lines)
        if new_content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except Exception as e:
                print(f"Error writing {file_path}: {e}", file=sys.stderr)
                return {}
    
    return fixes


def main():
    """Main function to find all broken links."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and fix broken links in markdown files")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix broken links")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    args = parser.parse_args()
    
    # Get the docs directory relative to this script
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / 'docs'
    
    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.fix or args.dry_run:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Fixing broken links in: {docs_dir}")
    else:
        print(f"Scanning for broken links in: {docs_dir}")
    print("=" * 50)
    
    markdown_files = find_markdown_files(str(docs_dir))
    total_broken_links = 0
    files_with_broken_links = 0
    
    # Count broken links by reason
    reason_counts = {
        "relative_path": 0,
        "wrong_prefix": 0, 
        "file_not_found": 0,
        "needs_lowercase": 0,
        "unknown": 0
    }
    
    # Count fixes if in fix mode
    total_fixes = {
        "examples_converted": 0,
        "framework_converted": 0,
        "workflows_converted": 0,
        "lowercased": 0,
        "failed_conversions": 0
    }
    
    for file_path in markdown_files:
        if args.fix or args.dry_run:
            # Fix mode: attempt to fix broken links
            fixes = fix_links_in_file(file_path, docs_dir, dry_run=args.dry_run)
            
            if any(fixes.values()):
                relative_path = Path(file_path).relative_to(docs_dir)
                print(f"\n📄 {relative_path}")
                print("-" * len(str(relative_path)))
                
                if fixes["examples_converted"] > 0:
                    print(f"  ✅ Fixed {fixes['examples_converted']} examples links")
                    total_fixes["examples_converted"] += fixes["examples_converted"]
                
                if fixes["framework_converted"] > 0:
                    print(f"  ✅ Fixed {fixes['framework_converted']} framework links")
                    total_fixes["framework_converted"] += fixes["framework_converted"]
                
                if fixes["workflows_converted"] > 0:
                    print(f"  ✅ Fixed {fixes['workflows_converted']} workflows links")
                    total_fixes["workflows_converted"] += fixes["workflows_converted"]
                
                if fixes["lowercased"] > 0:
                    print(f"  ✅ Lowercased {fixes['lowercased']} URLs")
                    total_fixes["lowercased"] += fixes["lowercased"]
                
                if fixes["failed_conversions"] > 0:
                    print(f"  ❌ Failed to fix {fixes['failed_conversions']} links")
                    total_fixes["failed_conversions"] += fixes["failed_conversions"]
        else:
            # Scan mode: just report broken links
            broken_links = check_file_for_broken_links(file_path, docs_dir)
            
            if broken_links:
                files_with_broken_links += 1
                # Make path relative to docs dir for cleaner output
                relative_path = Path(file_path).relative_to(docs_dir)
                print(f"\n📄 {relative_path}")
                print("-" * len(str(relative_path)))
                
                for url, line_num, reason in broken_links:
                    reason_icon = {
                        "relative_path": "🔗",
                        "wrong_prefix": "❌", 
                        "file_not_found": "📄",
                        "needs_lowercase": "🔤",
                        "unknown": "❓"
                    }.get(reason, "❓")
                    
                    print(f"  Line {line_num:3d}: {reason_icon} {url} ({reason})")
                    total_broken_links += 1
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print("\n" + "=" * 50)
    
    if args.fix or args.dry_run:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Fix Summary:")
        print(f"  Total markdown files processed: {len(markdown_files)}")
        print(f"  Examples links fixed: {total_fixes['examples_converted']}")
        print(f"  Framework links fixed: {total_fixes['framework_converted']}")
        print(f"  Workflows links fixed: {total_fixes['workflows_converted']}")
        print(f"  URLs lowercased: {total_fixes['lowercased']}")
        print(f"  Failed conversions: {total_fixes['failed_conversions']}")
        total_fixed = total_fixes['examples_converted'] + total_fixes['framework_converted'] + total_fixes['workflows_converted'] + total_fixes['lowercased']
        print(f"  Total links fixed: {total_fixed}")
    else:
        print(f"Scan Summary:")
        print(f"  Total markdown files scanned: {len(markdown_files)}")
        print(f"  Files with broken links: {files_with_broken_links}")
        print(f"  Total broken links found: {total_broken_links}")
        print(f"\nBreakdown by reason:")
        print(f"  🔗 Relative paths: {reason_counts['relative_path']}")
        print(f"  ❌ Wrong URL prefix: {reason_counts['wrong_prefix']}")
        print(f"  📄 File not found: {reason_counts['file_not_found']}")
        print(f"  🔤 Needs lowercase: {reason_counts['needs_lowercase']}")
        print(f"  ❓ Unknown: {reason_counts['unknown']}")
        
        if reason_counts['relative_path'] > 0:
            print(f"\nTo fix relative path links, run: python3 {sys.argv[0]} --fix")


if __name__ == "__main__":
    main()
