#!/bin/bash

# This function recursively searches for Python files starting from a given directory.
# Once it finds a directory with Python files and creates a py.typed file, it stops descending further into that path.
recursive_search_and_create() {
    local dir=$1
    local found_py_files=false

    # Check for Python files in the current directory
    if ls "$dir"/*.py &>/dev/null; then
        found_py_files=true
        # Only create py.typed if it doesn't already exist
        if [[ ! -f "$dir/py.typed" ]]; then
            echo "Creating py.typed in $dir"
            touch "$dir/py.typed"
        fi
    fi

    # If Python files were found, stop recursion for this path
    if $found_py_files; then
        return
    fi

    # Recursively search in subdirectories
    for subdir in "$dir"/*/; do
        if [[ -d $subdir ]]; then
            recursive_search_and_create "$subdir"
        fi
    done
}

# Main execution starts here
# Find all pyproject.toml files and then search for the llama_index directory from each location
find . -type f -name "pyproject.toml" | while read -r toml_file; do
    echo "Found pyproject.toml: $toml_file"
    base_dir=$(dirname "$toml_file")
    llama_index_dir="$base_dir/llama_index"

    if [ -d "$llama_index_dir" ]; then
        echo "Found llama_index directory: $llama_index_dir"
        recursive_search_and_create "$llama_index_dir"
    else
        echo "llama_index directory not found within $(dirname "$toml_file")"
    fi
done
