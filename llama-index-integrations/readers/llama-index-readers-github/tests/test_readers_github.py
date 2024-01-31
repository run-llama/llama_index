from llama_index.core.readers.base import BaseReader
from llama_index.readers.github import (
    GithubRepositoryReader,
    GitHubRepositoryCollaboratorsReader,
    GitHubRepositoryIssuesReader,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in GithubRepositoryReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [
        b.__name__ for b in GitHubRepositoryCollaboratorsReader.__mro__
    ]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in GitHubRepositoryIssuesReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
