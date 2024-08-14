# LlamaIndex Embeddings Integration: Vertex

Implements Vertex AI Embeddings Models:

| Model                                | Release Date      |
| ------------------------------------ | ----------------- |
| textembedding-gecko@003              | December 12, 2023 |
| textembedding-gecko@002              | November 2, 2023  |
| textembedding-gecko-multilingual@001 | November 2, 2023  |
| textembedding-gecko@001              | June 7, 2023      |
| multimodalembedding                  |                   |

**Note**: Currently Vertex AI does not support async on `multimodalembedding`.
Otherwise, `VertexTextEmbedding` supports async interface.

---

## **Version: [0.1.1]**

### **Key Enhancements**

1. **Flexible Credential Handling**:

   - Added `_process_credentials` to support credentials as JSON strings, dictionaries, or `service_account.Credentials` instances.

2. **Task Type Compatibility**:

   - Improved `_get_embedding_request` to automatically omit the `task_type` parameter for models like `textembedding-gecko@001`.

3. **Additional Configuration Options**:

   - Introduced support for `num_workers` in `VertexTextEmbedding` for better customization.

4. **Improved Initialization**:
   - Updated `init_vertexai` to utilize the new credential processing for seamless setup.
