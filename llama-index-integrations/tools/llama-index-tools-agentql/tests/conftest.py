def get_testing_data() -> dict:
    return {
        "TEST_DATA": {
            "product": {
                "name": "iKey FT-88-TP-USB Police Emergency Car Mount Backlit Red USB Keyboard B22",
                "price": 27.95,
                "condition": "Used",
            },
            "seller": {"name": "Ativo"},
        },
        "TEST_URL": "https://storage.googleapis.com/tf-benchmark/ebay_product_page/page.html",
        "TEST_QUERY": "{ product { name price condition } seller { name }}",
    }
