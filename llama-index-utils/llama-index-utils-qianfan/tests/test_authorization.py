from llama_index.utils.qianfan.authorization import (
    encode_canonical_query,
    encode_canonical_headers,
)


def test_encode_canonical_query() -> None:
    assert (
        encode_canonical_query("text&text1=测试&text10=test")
        == "text10=test&text1=%E6%B5%8B%E8%AF%95&text="
    )


def test_encode_canonical_headers() -> None:
    headers = {"Content-Type": "application/json"}
    host = "aip.baidubce.com"
    assert encode_canonical_headers(headers, host) == (
        "content-type;host",
        "content-type:application%2Fjson\nhost:aip.baidubce.com",
    )
