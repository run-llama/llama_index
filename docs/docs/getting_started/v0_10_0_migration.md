# Updating to v0.10.0

With the introduction of LlamaIndex v0.10.0, there were several changes

- integrations have separate `pip install`s (See the [full registry](https://llamahub.ai/))
- many imports changed
- the `ServiceContext` was deprecated

Thankfully, we've tried to make these changes as easy as possible!

## Migrating Imports

### Option 1: Use temporary legacy imports

Since this is such a large change, we have also provided a `legacy` import package so that existing code can migrate to v0.10.0 with minimal impact.

Using find+replace, you can update your imports from:

```python
from llama_index import VectorStoreIndex
from llama_index.llms import Ollama

...
```

to:

```python
from llama_index.legacy import VectorStoreIndex
from llama_index.legacy.llms import Ollama

...
```

### Option 2: Full migration

To help assist with migrating, `pip install llama-index` and `pip install llama-index-core` both come with a command-line tool to update existing code and notebooks.

**NOTE:** The CLI tool updates files in place. Please ensure you have your data backed up to undo any changes as needed.

After installing v0.10.0, you can upgrade your existing imports automatically:

```
llamaindex-cli upgrade-file <file_path>
# OR
llamaindex-cli upgrade <folder_path>
```

For notebooks, new `pip install` statements are inserted and imports are updated.

For `.py` and `.md` files, import statements are also updated, and new requirements are printed to the terminal.

## Deprecated ServiceContext

In addition to import changes, the existing `ServiceContext` has been deprecated. While it will be supported for a limited time, the preferred way of setting up the same options will be either globally in the `Settings` object or locally in the APIs that use certain modules.

For example, before you might have had:

```
from llama_index import ServiceContext, set_global_service_context

service_context = ServiceContext.from_defaults(
  llm=llm, embed_model=embed_model, chunk_size=512
)
set_global_service_context(service_context)
```

Which now looks like:

```
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

You can see the `ServiceContext` -> `Settings` migration guide for [more details](../module_guides/supporting_modules/service_context_migration.md).
