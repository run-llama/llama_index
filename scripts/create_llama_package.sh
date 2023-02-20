# create llama_index package

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

rm -rf $LLAMA_INDEX_DIR
mkdir -p $LLAMA_INDEX_DIR
# copy files from gpt_index dir
cp -r $GPT_INDEX_DIR/gpt_index $LLAMA_INDEX_DIR/llama_index
cp $GPT_INDEX_DIR/setup_llama.py $LLAMA_INDEX_DIR/setup.py
cp $GPT_INDEX_DIR/README.md $LLAMA_INDEX_DIR/README.md
cp $GPT_INDEX_DIR/LICENSE $LLAMA_INDEX_DIR/LICENSE
cp $GPT_INDEX_DIR/MANIFEST_llama.in $LLAMA_INDEX_DIR/MANIFEST.in
cd $LLAMA_INDEX_DIR
# replace all usages of gpt_index with llama_index
find llama_index/. -type f -name '*.py' -print0 | xargs -0 sed -i '' "s/gpt_index/llama_index/g"
# build package
python setup.py sdist