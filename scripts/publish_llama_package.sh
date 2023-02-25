#!/bin/bash

# publish llama_index package
if [[ -n "$LLAMA_INDEX_DIR" ]]; then
    echo "LLAMA_INDEX_DIR is set to $LLAMA_INDEX_DIR"
else
    echo "LLAMA_INDEX_DIR is not set"
    exit 1
fi

if [[ -n "$GPT_INDEX_DIR" ]]; then
    echo "GPT_INDEX_DIR is set to $GPT_INDEX_DIR"
else
    echo "GPT_INDEX_DIR is not set"
    exit 1
fi

if [[ -n "$PYPI_USERNAME" ]]; then
    echo "PYPI_USERNAME is set to $PYPI_USERNAME"
else
    echo "PYPI_USERNAME is not set"
    exit 1
fi

if [[ -n "$PYPI_PASSWORD" ]]; then
    echo "PYPI_PASSWORD is set to $PYPI_PASSWORD"
else
    echo "PYPI_PASSWORD is not set"
    exit 1
fi

LLAMA_INDEX_DIR=$LLAMA_INDEX_DIR GPT_INDEX_DIR=$GPT_INDEX_DIR sh $GPT_INDEX_DIR/scripts/create_llama_package.sh

# publish llama_index package
twine upload $LLAMA_INDEX_DIR/dist/* -u $PYPI_USERNAME -p $PYPI_PASSWORD

