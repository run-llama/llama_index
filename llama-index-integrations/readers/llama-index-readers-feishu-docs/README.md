# Feishu Doc Loader

```bash
pip install llama-index-readers-feishu-docs
```

This loader takes in IDs of Feishu Docs and parses their text into `documents`. You can extract a Feishu Doc's ID directly from its URL. For example, the ID of `https://test-csl481dfkgqf.feishu.cn/docx/HIH2dHv21ox9kVxjRuwc1W0jnkf` is `HIH2dHv21ox9kVxjRuwc1W0jnkf`. As a prerequisite, you will need to register with Feishu and build an custom app. See [here](https://open.feishu.cn/document/home/introduction-to-custom-app-development/self-built-application-development-process) for instructions.

## Usage

To use this loader, you simply need to pass in an array of Feishu Doc IDs. The default API endpoints are for Feishu, in order to switch to Lark, we should use `set_lark_domain`.

```python
app_id = "cli_slkdjalasdkjasd"
app_secret = "dskLLdkasdjlasdKK"
doc_ids = ["HIH2dHv21ox9kVxjRuwc1W0jnkf"]
from llama_index.readers.feishu_docs import FeishuDocsReader

loader = FeishuDocsReader(app_id, app_secret)
documents = loader.load_data(document_ids=doc_ids)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
