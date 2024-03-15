from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings, PromptTemplate
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ocigenai import OCIGenAI
from llama_index.embeddings.ocigenai import OCIGenAIEmbeddings
import faiss

documents = [
        "Larry Ellison co-founded Oracle Corporation in 1977 with Bob Miner and Ed Oates.",
        "Oracle Corporation is an American multinational computer technology company headquartered in Austin, Texas, United States.",
    ]

documents = [Document(text=doc) for doc in documents]

llm = OCIGenAI(
        model="cohere.command", # "meta.llama-2-70b-chat" or "cohere.command"
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.tenancy.oc1..aaaaaaaasz6cicsgfbqh6tj3xahi4ozoescfz36bjm3kucc7lotk2oqep47q",
        temperature=0.0,
        max_tokens=512,
        )

embed_model = OCIGenAIEmbeddings(
        model="cohere.embed-english-light-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.tenancy.oc1..aaaaaaaasz6cicsgfbqh6tj3xahi4ozoescfz36bjm3kucc7lotk2oqep47q",
        )


qa_prompt_template = """
Answer the question based only on the following context:
{context}
 
Question: {question}
"""
qa_prompt = PromptTemplate(template=qa_prompt_template, template_var_mappings={"context_str": "context", "query_str": "question"})


vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(384))
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)


query_engine = index.as_query_engine(llm=llm, text_qa_template=qa_prompt)

print(query_engine.query("when was oracle founded?"))
print(query_engine.query("where is oracle headquartered?"))
