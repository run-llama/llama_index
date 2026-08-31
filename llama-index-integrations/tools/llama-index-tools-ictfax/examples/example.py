"""Minimal ICTFax tool example."""
from llama_index.tools.ictfax import ICTFaxToolSpec

spec = ICTFaxToolSpec(
    base_url="https://fax.example.com/api", username="admin", password="mysecret"
)
doc = spec.upload_fax_document("invoice.pdf", name="April invoice")
print(doc)  # -> "Uploaded document id N"
print(spec.send_fax(to_number="+12025550123", document_id=12, title="April invoice"))
print(spec.get_fax_status(1))
