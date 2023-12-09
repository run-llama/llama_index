from llama_index import download_loader


def test_download_loader_do_not_crash_on_missing_init() -> None:
    download_loader("GithubRepositoryReader")
    assert True
