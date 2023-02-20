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

if [[ -n "$USERNAME" ]]; then
    echo "USERNAME is set to $USERNAME"
else
    echo "USERNAME is not set"
    exit 1
fi

if [[ -n "$PASSWORD" ]]; then
    echo "PASSWORD is set to $PASSWORD"
else
    echo "PASSWORD is not set"
    exit 1
fi

LLAMA_INDEX_DIR=$LLAMA_INDEX_DIR GPT_INDEX_DIR=$GPT_INDEX_DIR sh $GPT_INDEX_DIR/scripts/create_llama_package.sh

# publish llama_index package
twine upload $LLAMA_INDEX_DIR/dist/* -u $USERNAME -p $PASSWORD

