load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "perplexity_tool",
    srcs = [
        "base.py",
        "__init__.py",
    ],
    deps = [
        "//llama_index/core/tools/tool_spec:base",  # Dependency on the BaseToolSpec.
        "@pypi__requests//:pkg",                   # External dependency on requests.
    ],
)

py_test(
    name = "perplexity_tool_test",
    srcs = ["perplexity_tool_test.py"],
    deps = [
        ":perplexity_tool",
        "//llama_index/core/tools/tool_spec:base",
        "@pypi__requests//:pkg",
    ],
)