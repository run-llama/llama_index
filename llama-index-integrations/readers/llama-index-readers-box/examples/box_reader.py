import os
from typing import List
import dotenv

from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient, File
from llama_index.readers.box import (
    BoxReader,
    BoxReaderTextExtraction,
    BoxReaderAIPrompt,
    BoxReaderAIExtract,
)
from llama_index.core.schema import Document


def get_box_client() -> BoxClient:
    dotenv.load_dotenv()

    # Common configurations
    client_id = os.getenv("BOX_CLIENT_ID", "YOUR_BOX_CLIENT_ID")
    client_secret = os.getenv("BOX_CLIENT_SECRET", "YOUR_BOX_CLIENT_SECRET")

    # CCG configurations
    enterprise_id = os.getenv("BOX_ENTERPRISE_ID", "YOUR_BOX_ENTERPRISE_ID")
    ccg_user_id = os.getenv("BOX_USER_ID")

    config = CCGConfig(
        client_id=client_id,
        client_secret=client_secret,
        enterprise_id=enterprise_id,
        user_id=ccg_user_id,
    )

    auth = BoxCCGAuth(config)
    if config.user_id:
        auth.with_user_subject(config.user_id)

    return BoxClient(auth)


def get_testing_data() -> dict:
    return {
        "disable_folder_tests": True,
        "test_folder_id": "273980493541",
        "test_doc_id": "1584054722303",
        "test_ppt_id": "1584056661506",
        "test_xls_id": "1584048916472",
        "test_pdf_id": "1584049890463",
        "test_json_id": "1584058432468",
        "test_csv_id": "1584054196674",
        "test_txt_waiver_id": "1514587167701",
        "test_folder_invoice_po_id": "261452450320",
        "test_folder_purchase_order_id": "261457585224",
        "test_txt_invoice_id": "1517629086517",
        "test_txt_po_id": "1517628697289",
    }


def print_docs(label: str, docs: List[Document]):
    print("------------------------------")
    print(f"{label}: {len(docs)} document(s)")

    for doc in docs:
        print("------------------------------")
        file = File.from_dict(doc.extra_info)
        print(f"File ID: {file.id}\nName: {file.name}\nSize: {file.size} bytes")
        # print("------------------------------")
        print(f"Text: {doc.text[:100]} ...")
    print("------------------------------\n\n\n")


def main():
    box_client = get_box_client()
    test_data = get_testing_data()

    reader = BoxReader(box_client=box_client)
    docs = reader.load_data(file_ids=[test_data["test_txt_invoice_id"]])
    print_docs("Box Simple Reader", docs)

    reader = BoxReaderTextExtraction(box_client=box_client)
    docs = reader.load_data(file_ids=[test_data["test_txt_waiver_id"]])
    print_docs("BoxReader Text Extraction", docs)

    reader = BoxReaderAIPrompt(box_client=box_client)
    docs = reader.load_data(
        file_ids=[test_data["test_txt_waiver_id"]], ai_prompt="summarize this document"
    )
    print_docs("Box Reader AI Prompt", docs)

    reader = BoxReaderAIExtract(box_client=box_client)
    docs = reader.load_data(
        file_ids=[test_data["test_txt_invoice_id"]],
        ai_prompt='{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}',
    )
    print_docs("BoxReader AI Extract", docs)

    docs = reader.load_data(
        folder_id=test_data["test_folder_purchase_order_id"],
        ai_prompt='{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}',
        is_recursive=True,
    )
    print_docs("BoxReader AI Extract", docs)


if __name__ == "__main__":
    main()
