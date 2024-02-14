from llama_index.llms.mymagic import MyMagicAI

resp = MyMagicAI(
    api_key="test______________________Vitai",
    question="Who is the CEO?",
    storage_provider="s3",
    bucket_name="vitali-mymagic",
    ).acomplete("Paul Graham is ")
print(resp)
