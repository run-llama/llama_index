from llama_index.embeddings.ocigenai import OCIGenAIEmbeddings

embedding = OCIGenAIEmbeddings(
        model="cohere.embed-english-light-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.tenancy.oc1..aaaaaaaasz6cicsgfbqh6tj3xahi4ozoescfz36bjm3kucc7lotk2oqep47q",
        #input_type="CLUSTERING"
        )

e1 = embedding.get_text_embedding("This is a test document")
print(e1[-5:])
print(len(e1))

e2 = embedding.get_query_embedding("This is a test document")
print(e2[-5:])
print(len(e1))

docs = ["This is a test document", "This is another test document"]
e3 = embedding.get_text_embedding_batch(docs)
print(e3)
print(len(e3))
