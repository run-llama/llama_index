# Required Environment Variables: OPENAI_API_KEY

from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
DocugamiKgRagPack = download_llama_pack("DocugamiKgRagPack", "./docugami_kg_rag")

# create the pack
pack = DocugamiKgRagPack()

# list the docsets in your Docugami organization and set the docset_id
pack.list_docset()
docset_id = "5bcy7abew0sd"

pack.index_docset(docset_id)
pack.build_agent_for_docset(docset_id, use_reports=True)

pack.run("What is the Early Bird Discount for a visit to Indonesia?")

# A query that uses the Docugami reports to find more accurate answers
pack.run("List all the early bird discounts available")
