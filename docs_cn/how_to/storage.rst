💾存储
============

LlamaIndex提供了一个高级接口，用于摄取、索引和查询您的外部数据。
默认情况下，LlamaIndex隐藏了复杂性，让您在`不到5行代码</how_to/storage/customization.html>`_中查询数据。

在幕后，LlamaIndex还支持可替换的**存储组件**，允许您自定义：

- **文档存储**：存储摄取的文档（即`Node`对象）的位置，
- **索引存储**：存储索引元数据的位置，
- **向量存储**：存储嵌入向量的位置。

文档/索引存储依赖于通用的键值存储抽象，详情参见下文。

LlamaIndex支持将数据持久化到`fsspec <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_支持的任何存储后端。
我们已经确认支持以下存储后端：

- 本地文件系统
- AWS S3
- Cloudflare R2

有关如何使用LlamaIndex与Cloudflare R2一起使用的示例，请参见`此示例</examples/vector_stores/SimpleIndexOnS3.html>`_。

.. image:: ../_static/storage/storage.png
   :class: only-light


.. toctree::
   :maxdepth: 1
   :caption: 存储

   storage/save_load.md
   storage/customization.md
   storage/docstores.md
   storage/index_stores.md
   storage/vector_stores.md
   storage/kv_stores.md

LlamaIndex提供了一个高级接口，用于摄取、索引和查询您的外部数据。默认情况下，LlamaIndex隐藏了复杂性，让您在不到5行代码中查询数据。在幕后，LlamaIndex还支持可替换的存储组件，允许您自定义文档存储（存储摄取的文档（即Node对象）的位置）、索引存储（存储索引元数据的位置）和向量存储（存储嵌入向量的位置）。LlamaIndex支持将数据持久化到fsspec支持的任何存储后端，包括本地文件系统、AWS S3和Cloudflare R2。