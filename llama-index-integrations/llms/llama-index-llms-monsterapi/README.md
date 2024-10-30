# LlamaIndex Llms Integration: Monsterapi

MonsterAPI LLM.

Monster Deploy enables you to host any vLLM supported large language model (LLM) like Tinyllama, Mixtral, Phi-2 etc as a rest API endpoint on MonsterAPI's cost optimised GPU cloud.

With MonsterAPI's integration in Llama index, you can use your deployed LLM API endpoints to create RAG system or RAG bot for use cases such as: - Answering questions on your documents - Improving the content of your documents - Finding context of importance in your documents

Once deployment is launched use the base_url and api_auth_token once deployment is live and use them below.

Note: When using LLama index to access Monster Deploy LLMs, you need to create a prompt with required template and send compiled prompt as input.

See `LLama Index Prompt Template Usage example` section for more details.

see (https://developer.monsterapi.ai/docs/monster-deploy-beta) for more details

Once deployment is launched use the base_url and api_auth_token once deployment is live and use them below.

Note: When using LLama index to access Monster Deploy LLMs, you need to create a prompt with reqhired template and send compiled prompt as input. see section `LLama Index Prompt Template
    Usage example` for more details.

Examples:

`pip install llama-index-llms-monsterapi`

1.  MonsterAPI Private LLM Deployment use case

    ````python
    from llama_index.llms.monsterapi import MonsterLLM

        llm = MonsterLLM(
            model = "<Replace with basemodel used to deploy>",
            api_base="https://ecc7deb6-26e0-419b-a7f2-0deb934af29a.monsterapi.ai",
            api_key="a0f8a6ba-c32f-4407-af0c-169f1915490c",
            temperature=0.75,
        )

        response = llm.complete("What is the capital of France?")
        ```

    ````

2.  Monster API General Available LLMs

    ````python3
    from llama_index.llms.monsterapi import MonsterLLM

        llm = MonsterLLM(model="microsoft/Phi-3-mini-4k-instruct")

        response = llm.complete("What is the capital of France?")
        print(str(response))
        ```
    ````
