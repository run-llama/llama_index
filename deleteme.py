# import logging
import os

from langchain import OpenAI

from gpt_index import GPTSimpleVectorIndex, LLMPredictor, QueryMode, download_loader
from gpt_index.readers.llamahub_modules import GithubClient, GithubRepositoryReader

# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# download_loader("GithubRepositoryReader")


assert (
    os.getenv("OPENAI_API_KEY") is not None
), "Please set the OPENAI_API_KEY environment variable."

import pickle

if __name__ == "__main__":
    docs = None
    # if os.path.exists("docs.pkl"):
    #     with open("docs.pkl", "rb") as f:
    #         docs = pickle.load(f)
    if docs is None:
        github_client = GithubClient()
        allowed_file_extensions = [".py", ".md", ".MD", ".pdf", ".rst"]
        allowed_directories = ["gpt_index", "docs"]
        loader = GithubRepositoryReader(
            github_client,
            owner="ahmetkca",
            repo="gpt_index",
            use_parser=True,
            filter_directories=(
                allowed_directories,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            filter_file_extensions=(
                allowed_file_extensions,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            verbose=True,
            concurrent_requests=10,
        )

        docs = loader.load_data(branch="main")

        with open("docs.pkl", "wb") as f:
            pickle.dump(docs, f)

    for doc in docs:
        assert any(
            doc.extra_info["file_path"].endswith(ext) for ext in allowed_file_extensions
        ), f"Forbidden file extension: {doc}, expected one of {allowed_file_extensions} but got {doc.split('.')[-1]}"
        print(doc.extra_info)

    exit()

    llm_predictor = LLMPredictor(
        OpenAI(max_tokens=256 * 2, verbose=True, cache=False, temperature=0.25)
    )

    index = None
    if os.path.exists("delme_index.json"):
        index = GPTSimpleVectorIndex.load_from_disk("delme_index.json")

    if index is None:
        index = GPTSimpleVectorIndex(
            docs,
            llm_predictor=llm_predictor,
        )

    index.save_to_disk("delme_index.json")

    query_str = "Explain how does the GPTSimpleVectorIndex work in detail"
    print("Similarity mode:")
    for similarity_top_k in range(1, 5):
        response = index.query(query_str, similarity_top_k=similarity_top_k)
        print(f"Similarity top {similarity_top_k}:\n{response}")

    print("Embedding mode:")
    for similarity_top_k in range(1, 5):
        response = index.query(
            query_str, similarity_top_k=similarity_top_k, mode=QueryMode.EMBEDDING
        )
        print(f"Embedding top {similarity_top_k}:\n{response}")
