# Feishu Wiki Loader

This loader can traverse all feishu documents under the feishi space.

## Usage

To use this loader, you need to:

1. apply the permission(`wiki:wiki:readonly`) of the feishu app
2. add the feishu app as the admin of your feishu space, see [here](https://open.feishu.cn/document/server-docs/docs/wiki-v2/wiki-qa#b5da330b) for more help
3. finally, pass your feishu space id to this loader

```python
app_id = "xxx"
app_secret = "xxx"
space_id = "xxx"
FeishuWikiReader = download_loader("FeishuWikiReader")
loader = FeishuWikiReader(app_id, app_secret)
documents = loader.load_data(space_id=space_id)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
