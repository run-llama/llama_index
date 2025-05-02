python_sources()

python_tests(
    name="tests",
    dependencies=[
        ":config",
        ":llama-index-readers-file",
    ],
)

files(
    name="config",
    sources=["data/**/*"]
)

python_requirement(
    name="llama-index-readers-file",
    requirements=["llama-index-readers-file"],
)
