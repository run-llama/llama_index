import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typing import Callable, Any, Union, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from .entities import (
    Article,
    ArticleSlim,
    Category,
    ProjectVersion,
)
from .errors import RateLimitException

BASE_URL = "https://apihub.document360.io/v2"


class Document360Reader(BaseReader):
    def __init__(
        self,
        api_key: str,
        should_process_project_version: Optional[
            Callable[[ProjectVersion], bool]
        ] = None,
        should_process_category: Optional[
            Callable[[Category, list[Category]], bool]
        ] = None,
        should_process_article: Optional[Callable[[ArticleSlim], bool]] = None,
        handle_batch_finished: Optional[Callable[[], Any]] = None,
        handle_rate_limit_error: Optional[Callable[[], Any]] = None,
        handle_request_http_error: Optional[
            Callable[[requests.exceptions.HTTPError], Any]
        ] = None,
        handle_category_processing_started: Optional[Callable[[Category], Any]] = None,
        handle_article_processing_started: Optional[Callable[[Article], Any]] = None,
        handle_article_processing_error: Optional[
            Callable[[Union[Article, ArticleSlim]], Any]
        ] = None,
        handle_load_data_error: Optional[Callable[[Exception, Article], Any]] = None,
        article_to_custom_document: Optional[Callable[[Article], Document]] = None,
        rate_limit_num_retries=10,
        rate_limit_retry_wait_time=30,
    ):
        self.api_key = api_key
        self.processed_articles = []
        self.should_process_project_version = should_process_project_version
        self.should_process_category = should_process_category
        self.should_process_article = should_process_article
        self.handle_batch_finished = handle_batch_finished
        self.handle_rate_limit_error = handle_rate_limit_error
        self.handle_request_http_error = handle_request_http_error
        self.handle_article_processing_error = handle_article_processing_error
        self.handle_category_processing_started = handle_category_processing_started
        self.handle_article_processing_started = handle_article_processing_started
        self.handle_load_data_error = handle_load_data_error
        self.headers = {"api_token": api_key, "Content-Type": "application/json"}
        self.article_to_custom_document = article_to_custom_document
        self.rate_limit_num_retries = rate_limit_num_retries
        self.rate_limit_retry_wait_time = rate_limit_retry_wait_time

        self._make_request = self._configure_request_retry(self._make_request)

    def _configure_request_retry(self, func):
        return retry(
            stop=stop_after_attempt(self.rate_limit_num_retries),
            wait=wait_fixed(self.rate_limit_retry_wait_time),
            retry=retry_if_exception_type(RateLimitException),
        )(func)

    def _make_request(
        self, method, url, headers=None, params=None, data=None, json=None
    ):
        response = requests.request(
            method, url, headers=headers, params=params, data=data, json=json
        )

        if response.status_code == 429:
            self.handle_rate_limit_error and self.handle_rate_limit_error()
            raise RateLimitException("Rate limit exceeded")

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.handle_request_http_error and self.handle_request_http_error(e)

            raise

        return response

    def _fetch_project_versions(self):
        url = f"{BASE_URL}/ProjectVersions"
        response = self._make_request("GET", url, headers=self.headers)
        return response.json()

    def _get_categories(self, project_version_id: str):
        url = f"{BASE_URL}/ProjectVersions/{project_version_id}/categories"
        response = self._make_request("GET", url, headers=self.headers)
        return response.json()

    def _fetch_article(self, articleId):
        url = f"{BASE_URL}/Articles/{articleId}"
        response = self._make_request("GET", url, headers=self.headers)
        return response.json()

    def _get_document360_response_data(self, response):
        return response["data"]

    def _process_category_recursively(self, category: Category, parent_categories=[]):
        if self.should_process_category and not self.should_process_category(
            category, parent_categories
        ):
            # we still might find the category of interest in the child categories
            # even if the current category is not of interest
            for child_category in category.child_categories:
                self._process_category_recursively(
                    child_category,
                    [*parent_categories, category],
                )
            return

        self.handle_category_processing_started and self.handle_category_processing_started(
            category
        )
        articles = category.articles
        for article_slim in articles:
            article = None

            try:
                if self.should_process_article and not self.should_process_article(
                    article_slim
                ):
                    continue

                article_response = self._fetch_article(article_slim.id)
                _article = self._get_document360_response_data(article_response)

                article = Article(
                    id=_article["id"],
                    title=_article["title"],
                    content=_article["content"],
                    html_content=_article["html_content"],
                    category_id=_article["category_id"],
                    project_version_id=_article["project_version_id"],
                    version_number=_article["version_number"],
                    public_version=_article["public_version"],
                    latest_version=_article["latest_version"],
                    enable_rtl=_article["enable_rtl"],
                    hidden=_article["hidden"],
                    status=_article["status"],
                    order=_article["order"],
                    created_by=_article["created_by"],
                    authors=_article["authors"],
                    created_at=_article["created_at"],
                    modified_at=_article["modified_at"],
                    slug=_article["slug"],
                    is_fall_back_content=_article["is_fall_back_content"],
                    description=_article["description"],
                    category_type=_article["category_type"],
                    content_type=_article["content_type"],
                    is_shared_article=_article["is_shared_article"],
                    translation_option=_article["translation_option"],
                    url=_article["url"],
                )

                self.handle_article_processing_started and self.handle_article_processing_started(
                    article
                )

                self.processed_articles.append(article)
            except Exception as e:
                self.handle_article_processing_error and self.handle_article_processing_error(
                    e, article or article_slim
                )

                continue

        for child_category in category.child_categories:
            self._process_category_recursively(
                child_category,
                [*parent_categories, category],
            )

    def _fetch_articles(self):
        project_versions_response = self._fetch_project_versions()
        project_versions = self._get_document360_response_data(
            project_versions_response
        )

        for _project_version in project_versions:
            project_version = ProjectVersion(
                id=_project_version["id"],
                version_number=_project_version["version_number"],
                base_version_number=_project_version["base_version_number"],
                version_code_name=_project_version["version_code_name"],
                is_main_version=_project_version["is_main_version"],
                is_beta=_project_version["is_beta"],
                is_public=_project_version["is_public"],
                is_deprecated=_project_version["is_deprecated"],
                created_at=_project_version["created_at"],
                modified_at=_project_version["modified_at"],
                language_versions=_project_version["language_versions"],
                slug=_project_version["slug"],
                order=_project_version["order"],
                version_type=_project_version["version_type"],
            )

            if (
                self.should_process_project_version
                and not self.should_process_project_version(project_version)
            ):
                continue

            categories_response = self._get_categories(project_version.id)
            categories = self._get_document360_response_data(categories_response)

            for _category in categories:
                category = Category(
                    id=_category["id"],
                    name=_category["name"],
                    child_categories=_category["child_categories"],
                    articles=_category["articles"],
                    description=_category["description"],
                    project_version_id=_category["project_version_id"],
                    order=_category["order"],
                    parent_category_id=_category["parent_category_id"],
                    hidden=_category["hidden"],
                    icon=_category["icon"],
                    slug=_category["slug"],
                    language_code=_category["language_code"],
                    category_type=_category["category_type"],
                    created_at=_category["created_at"],
                    modified_at=_category["modified_at"],
                    status=_category["status"],
                    content_type=_category["content_type"],
                )

                self._process_category_recursively(category)

        self.handle_batch_finished and self.handle_batch_finished()

        articles_collected = self.processed_articles.copy()
        self.processed_articles = []

        return articles_collected

    def article_to_document(self, article: Article):
        return Document(
            doc_id=article.id,
            text=article.html_content,
            extra_info={
                "title": article.title,
                "content": article.content,
                "category_id": article.category_id,
                "project_version_id": article.project_version_id,
                "version_number": article.version_number,
                "public_version": article.public_version,
                "latest_version": article.latest_version,
                "enable_rtl": article.enable_rtl,
                "hidden": article.hidden,
                "status": article.status,
                "order": article.order,
                "created_by": article.created_by,
                "authors": article.authors,
                "created_at": article.created_at,
                "modified_at": article.modified_at,
                "slug": article.slug,
                "is_fall_back_content": article.is_fall_back_content,
                "description": article.description,
                "category_type": article.category_type,
                "content_type": article.content_type,
                "is_shared_article": article.is_shared_article,
                "translation_option": article.translation_option,
                "url": article.url,
            },
        )

    def load_data(self):
        try:
            articles = self._fetch_articles()

            return list(
                map(
                    self.article_to_custom_document or self.article_to_document,
                    articles,
                )
            )
        except Exception as e:
            if not self.handle_load_data_error:
                raise

            self.handle_load_data_error(e, self.processed_articles)
