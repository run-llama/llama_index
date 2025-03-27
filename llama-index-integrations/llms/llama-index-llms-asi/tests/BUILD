load("@llama_index_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "python_tests")

package(default_visibility = ["//visibility:public"])

python_tests(
    name = "test_llms_asi",
    srcs = ["test_llms_asi.py"],
    deps = [
        "//llama_index/llms/asi",
        "//llama_index/core:llama_index_core",
        requirement("pytest"),
    ],
)
