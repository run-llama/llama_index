# LlamaPacks 📦

## Llama-Pack Usage

If you merely intend to use the llama-pack, then the recommended route is via pip install:

```
pip install llama-index-packs-<name-of-pack>
```

For a list of our llama-packs Python packages, visit [llamahub.ai](https://llamahub.ai/?tab=llama-packs).

On the other hand, if you wish to download a llama-pack and potentially customize it,
you can download it as a template. There are a couple of ways to do so. First,
llama-packs can be downloaded as a template by using the `llamaindex-cli` tool
that comes with `llama-index`:

```bash
llamaindex-cli download-llamapack ZephyrQueryEnginePack --download-dir ./zephyr_pack
```

Or with the `download_llama_pack` function directly (in this case, you must supply
a download directory):

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
LlavaCompletionPack = download_llama_pack(
    "LlavaCompletionPack", "./llava_pack"  # ./llava_pack is the download dir
)
```

## Features

- **Easy Integration:** Quickly add Llama-Packs to your projects with minimal setup.
- **Customizable Templates:** Download packs and modify them to suit your needs.
- **Rich Documentation:** Comprehensive guides and examples available online.

## Usage Examples

Here’s a simple example of using a downloaded Llama-Pack in your application:

```python
from llama_index import SomeClassFromPack

# Initialize and use the class from the downloaded pack
pack_instance = SomeClassFromPack()
result = pack_instance.perform_action()
print(result)
```

## Changelog

| Version | Description                                         |
|---------|-----------------------------------------------------|
| v1.0.0  | Initial release with basic features.                |
| v1.1.0  | Added more Llama-Packs to the repository.          |
|         | Improved documentation and usage examples.          |

