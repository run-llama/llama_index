# from llama_index.readers.mineru import MinerUReader

# reader = MinerUReader()
# print(f"Mode: {reader.mode}")

from llama_index.readers.mineru import MinerUReader

reader = MinerUReader()

# Parse a single PDF from URL
documents = reader.load_data("https://cdn-mineru.openxlab.org.cn/demo/example.pdf")
print(documents[0].text)
