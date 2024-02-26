CATEGORIZER_PROMPT = """

You are an AI tool meant to categorize questions into a set of curated categories.

Here are the categories and additional information for each category you're meant to categorize the question below into: (Examples may be separated by a ;) \n
{category_info}

When responding with the selected category above, please ONLY respond with the category.
Do not respond with your explanation or context, your response must only contain a category above.

Please categorize this question into one of the categories above:\n
{question}

""".strip()

DEFAULT_CATEGORIES = {  # key, #alpha, description [and examples]
    "web search query": {
        "alpha": 0,
        "description": "Shortened queries similar to those commonly entered into a search engine (often an incomplete sentence)",
        "examples": [
            "Transfer capabilities of LLaMA language model to non-English languages",
            "Best retrieval concept queries",
        ],
    },
    "concept seeking query": {
        "alpha": 0.2,
        "description": "Abstract questions, usually on a specific topic, that require multiple sentences to answer",
        "examples": [
            "What is the dual-encoder architecture used in recent works on dense retrievers?",
            "Why should I use semantic search to rank results?",
        ],
    },
    "fact seeking query": {
        "alpha": 0.6,
        "description": "Queries with a single, clear answer",
        "examples": [
            "What is the total number of propositions the English Wikipedia dump is segmented into in FACTOID WIKI?",
            "How many documents are semantically ranked?",
        ],
    },
    "keyword query": {
        "alpha": 0.4,
        "description": "Short queries that consist of only the important identifier words.",
        "examples": ["GTR retriever recall rate", "semantic ranker"],
    },
    "queries with misspellings": {
        "alpha": 1,
        "description": "Queries with typos, transpositions and common misspellings introduced",
        "examples": [
            "What is the advntage of prposition retrieval over sentnce or passage retrieval?",
            "Ho w mny documents are samantically r4nked",
        ],
    },
    "exact sub-string searches": {
        "alpha": 0,
        "description": "Longer queries that are exact sub-strings from the original paragraph",
        "examples": [
            "first kwords for the GTR retriever. Finer-grained",
            "enables you to maximize the quality and value of your LLM investments most efficiently by feeding only relevant information",
        ],
    },
}
