Based on the provided commit diff, **the following `llama-index-integrations/llms` packages have their handling of the `tool_required` parameter changed, and therefore need to be tested for correct support of the parameter**. Packages that do not make use of (or support) the `tool_required` parameter (i.e., where it is ignored or explicitly documented as unsupported) **do not count as changed and can be ignored for verification**.

### Packages that NEED testing for `tool_required`

The following packages now use the `tool_required` parameter to influence their function-calling/tool-calling behavior, and should be tested to verify correct handling:

1. **llama-index-llms-anthropic**

   - Now directly maps `tool_required` to Anthropic’s `"tool_choice"` (`"any"` if required, `"auto"` otherwise).

2. **llama-index-llms-azure-inference**

   - `tool_required` is mapped to `"tool_choice"` using Azure's enums; `"REQUIRED"` if required, `"AUTO"` otherwise.

3. **llama-index-llms-bedrock-converse**

   - Uses `tool_required` to set `"toolChoice"` (now always injected as either `{"auto":{}}` or `{"any":{}}`).

4. **llama-index-llms-cohere**

   - `"tool_choice": "REQUIRED"` if required, omitted otherwise.

5. **llama-index-llms-gemini**

   - Maps `tool_required` to the function-calling config (`"mode": FunctionCallingMode.ANY if required else AUTO`).

6. **llama-index-llms-huggingface-api**

   - `"tool_choice": "required"` if `tool_required`, `"auto"` otherwise.

7. **llama-index-llms-ibm**

   - If `tool_required` and no specific tool set, selects the first available tool; sets `"tool_choice"` accordingly.

8. **llama-index-llms-litellm**

   - `"tool_choice": "required"` if `tool_required`, `"auto"` otherwise.

9. **llama-index-llms-mistralai**

   - `"tool_choice": "required"` if `tool_required`, `"auto"` otherwise.

10. **llama-index-llms-oci-data-science**

    - If `tool_required`, injects `"required"` string as `tool_choice` (unless a custom one is supplied).

11. **llama-index-llms-oci-genai**

    - `"tool_choice": "REQUIRED"` if `tool_required`, omitted otherwise.

12. **llama-index-llms-openai**
    - Passes `tool_required` to `resolve_tool_choice`; if no specific tool provided, sets to `"required"`.

#### Summary Table

| Package          | `tool_required` Influences Payload/Logic     |
| ---------------- | -------------------------------------------- |
| ai21             | **NO** (ignored, comment says not supported) |
| anthropic        | YES                                          |
| azure-inference  | YES                                          |
| bedrock-converse | YES                                          |
| cohere           | YES                                          |
| dashscope        | **NO** (documented as not supported)         |
| deepinfra        | **NO** (not supported per docs)              |
| gemini           | YES                                          |
| huggingface-api  | YES                                          |
| ibm              | YES                                          |
| litellm          | YES                                          |
| mistralai        | YES                                          |
| oci-data-science | YES                                          |
| oci-genai        | YES                                          |
| ollama           | **NO** (“tool_required” present but unused)  |
| openai           | YES                                          |
| siliconflow      | **NO** (not supported per docs)              |
| vertex           | **NO** (“tool_required” present but unused)  |
| zhipuai          | **NO** (documented as not supported)         |

### **In summary, you need to test:**

> - anthropic
> - azure-inference
> - bedrock-converse
> - cohere
> - gemini
> - huggingface-api
> - ibm
> - litellm
> - mistralai
> - oci-data-science
> - oci-genai
> - openai

**The remaining LLM integrations can be ignored for this verification (either `tool_required` is not supported, or the parameter is present but unused).**
