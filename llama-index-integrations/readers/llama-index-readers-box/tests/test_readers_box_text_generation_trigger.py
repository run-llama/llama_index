from types import SimpleNamespace

from llama_index.readers.box.BoxAPI import box_api


def _pending_representation():
    """An `extracted_text` representation Box has not generated yet."""
    return SimpleNamespace(
        representation="extracted_text",
        status=SimpleNamespace(state="none"),
        info=SimpleNamespace(url="https://api.box.com/2.0/files/1/content/text.txt"),
        content=SimpleNamespace(url_template="https://dl.boxcloud.com/{+asset_path}"),
    )


def test_pending_text_representation_is_triggered_with_the_box_client(monkeypatch):
    """
    `_do_request` takes (box_client, url).

    The call that triggers generation for a still-pending representation must pass
    the client too, exactly like the download call right below it.
    """
    calls = []

    def fake_do_request(box_client, url):
        calls.append((box_client, url))
        return b"CONTENT"

    monkeypatch.setattr(box_api, "_do_request", fake_do_request)

    entry = _pending_representation()
    representation_file = SimpleNamespace(
        id="1", representations=SimpleNamespace(entries=[entry])
    )
    box_client = SimpleNamespace(
        files=SimpleNamespace(
            get_file_by_id=lambda *args, **kwargs: representation_file
        )
    )
    box_file = SimpleNamespace(id="1", text_representation=None)

    result = box_api.get_text_representation(box_client, [box_file], token_limit=4)

    # both the trigger and the download go through the client
    assert [client for client, _ in calls] == [box_client, box_client]
    assert calls[0][1] == entry.info.url
    assert calls[1][1] == "https://dl.boxcloud.com/"

    assert result == [box_file]
    assert box_file.text_representation == b"CONT"
