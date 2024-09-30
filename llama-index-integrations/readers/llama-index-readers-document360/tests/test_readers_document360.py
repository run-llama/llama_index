import pytest
from unittest.mock import patch, MagicMock
from llama_index.readers.document360 import Document360Reader

from llama_index.readers.document360.errors import (
    RetryError,
    HTTPError,
)


def test_document360reader_initialization():
    api_key = "test_api_key"
    Document360Reader(api_key=api_key)


@patch("requests.request")
def test_make_request_rate_limit(mock_request):
    reader = Document360Reader(
        api_key="test_api_key", rate_limit_retry_wait_time=0, rate_limit_num_retries=0
    )

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_request.return_value = mock_response

    with pytest.raises(RetryError):
        reader._make_request("GET", "some_url")


@patch("requests.request")
def test_make_request_http_error(mock_request):
    reader = Document360Reader(api_key="test_api_key")

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.raise_for_status.side_effect = HTTPError("HTTP error")
    mock_request.return_value = mock_response

    with pytest.raises(HTTPError):
        reader._make_request("GET", "some_url")


@patch("requests.request")
def test_make_request_retries(mock_request):
    reader = Document360Reader(
        api_key="test_api_key", rate_limit_num_retries=3, rate_limit_retry_wait_time=0
    )

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_request.return_value = mock_response

    with pytest.raises(RetryError):
        reader._make_request("GET", "some_url")

    assert mock_request.call_count == 3


@patch("requests.request")
def test_fetch_project_versions(mock_request):
    reader = Document360Reader(api_key="test_api_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "version1"}]}
    mock_request.return_value = mock_response

    result = reader._fetch_project_versions()

    assert result == {"data": [{"id": "version1"}]}
    assert mock_request.call_count == 1


@patch("requests.request")
def test_fetch_article(mock_request):
    reader = Document360Reader(api_key="test_api_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {"id": "article1", "content": "Test content"}
    }
    mock_request.return_value = mock_response

    result = reader._fetch_article("article1")

    assert result == {"data": {"id": "article1", "content": "Test content"}}
    assert mock_request.call_count == 1


@patch("requests.request")
def test_load_data(mock_request):
    reader = Document360Reader(api_key="test_api_key")

    project_versions_mock_response = {
        "data": [
            {
                "id": "23ff5ba3-9c74-4e2f-8215-88e91596b4c1",
                "version_number": 1.0,
                "base_version_number": 0.0,
                "version_code_name": "Project Version",
                "is_main_version": True,
                "is_beta": False,
                "is_public": True,
                "is_deprecated": False,
                "created_at": "2022-03-09T09:12:14.21Z",
                "modified_at": "2022-03-09T09:41:03.062Z",
                "language_versions": [
                    {
                        "id": "5d379e06-3678-4899-a756-c5a9a145289f",
                        "name": "English",
                        "code": "en",
                        "set_as_default": True,
                        "hidden": False,
                        "enable_rtl": False,
                        "site_protection_level": 1,
                        "is_inheritance_disabled": False,
                        "has_inheritance_disabled_categories_or_articles": False,
                        "country_flag_code": None,
                        "is_home_page_enabled": False,
                        "version_display_name": None,
                    }
                ],
                "slug": "v1",
                "order": 0,
                "version_type": 0,
            },
        ],
        "extension_data": None,
        "success": True,
        "errors": [],
        "warnings": [],
        "information": [],
    }

    mock_response_project_versions = MagicMock()
    mock_response_project_versions.status_code = 200
    mock_response_project_versions.json.return_value = project_versions_mock_response

    categories_mock_response = {
        "data": [
            {
                "id": "13a259b8-f91a-41e5-a0a5-781f2240d0b1",
                "name": "Category",
                "description": None,
                "project_version_id": "23ff5ba3-9c74-4e2f-8215-88e91596b4c1",
                "order": 2,
                "parent_category_id": None,
                "hidden": False,
                "articles": [
                    {
                        "id": "a87c76c4-ef0c-4f3e-a65c-56935192481b",
                        "title": "Article",
                        "public_version": 2,
                        "latest_version": 2,
                        "language_code": "en",
                        "hidden": False,
                        "status": 3,
                        "order": 1,
                        "slug": "article-name",
                        "content_type": 0,
                        "translation_option": 0,
                        "is_shared_article": False,
                        "modified_at": "2023-09-05T23:16:56.376Z",
                    }
                ],
                "child_categories": [],
                "icon": None,
                "slug": "drafts",
                "language_code": "en",
                "category_type": 0,
                "created_at": "2023-08-29T22:18:41.607Z",
                "modified_at": "2023-08-29T22:18:41.607Z",
                "status": None,
                "content_type": None,
            }
        ]
    }

    mock_response_categories = MagicMock()
    mock_response_categories.status_code = 200
    mock_response_categories.json.return_value = categories_mock_response

    article_mock_response = {
        "data": {
            "id": "a87c76c4-ef0c-4f3e-a65c-56935192481b",
            "title": "Article",
            "content": "## Content",
            "html_content": "<h2>Content</h2>",
            "category_id": "bb50902e-cae3-41e5-8deb-43d3c2425060",
            "project_version_id": "23ff5ba3-9c74-4e2f-8215-88e91596b4c1",
            "version_number": 2,
            "public_version": 2,
            "latest_version": 2,
            "enable_rtl": False,
            "hidden": False,
            "status": 3,
            "order": 1,
            "created_by": "3513e14e-02d3-4e17-8c20-5af5ff832a1b",
            "authors": [
                {
                    "id": "3513e14e-02d3-4e17-8c20-5af5ff832a1b",
                    "first_name": "First",
                    "last_name": "Last",
                    "user_description": None,
                    "unique_user_name": "first-last",
                    "email_id": "first-last@test.com",
                    "profile_logo_url": "https://example.com/profile.jpg",
                    "profile_logo_cdn_url": "https://example.com/profile.jpg",
                    "is_enterprise_user": False,
                }
            ],
            "created_at": "2023-09-05T19:20:22.889Z",
            "modified_at": "2023-09-05T23:16:56.361Z",
            "slug": "article-name",
            "is_fall_back_content": False,
            "description": None,
            "category_type": 0,
            "content_type": 0,
            "is_shared_article": False,
            "translation_option": 0,
            "url": "https://example.com/article",
        },
        "extension_data": None,
        "success": True,
        "errors": [],
        "warnings": [],
        "information": [],
    }

    mock_response_article = MagicMock()
    mock_response_article.status_code = 200
    mock_response_article.json.return_value = article_mock_response

    mock_request.side_effect = [
        mock_response_project_versions,
        mock_response_categories,
        mock_response_article,
    ]

    result = reader.load_data()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].text == "<h2>Content</h2>"
    assert result[0].doc_id == "a87c76c4-ef0c-4f3e-a65c-56935192481b"
