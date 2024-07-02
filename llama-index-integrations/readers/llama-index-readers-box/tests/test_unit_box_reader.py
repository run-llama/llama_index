from llama_index.readers.box.base import BoxReader


def test_box_reader_connect_config(box_config):
    reader = BoxReader(
        box_client_id=box_config.client_id, box_client_secret=box_config.client_secret
    )
    reader.load_data()


# def test_box_reader_connect_client(box_client):
#     reader = BoxReader(box_client)
