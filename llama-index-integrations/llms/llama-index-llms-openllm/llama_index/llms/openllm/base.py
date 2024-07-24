from llama_index.llms.openai.base import OpenAI


class OpenLLM(OpenAI):
    """
    OpenLLM LLM.

    A thin wrapper around OpenAI interface to help users interact with OpenLLM's running server.


    Examples:
        `pip install llama-index-llms-openllm`

        ```python
        from llama_index.llms.openllm import OpenLLM

        llm = OpenLLM(model="my-model", api_base="https://hostname.com/v1", api_key="na")

        stream = llm.stream("Hi, write a short story")

        for r in stream:
            print(r.delta, end="")
        ```
    """

    @classmethod
    def class_name(cls) -> str:
        return "OpenLLM"
