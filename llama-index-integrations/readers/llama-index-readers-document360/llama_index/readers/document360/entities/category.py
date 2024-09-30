from .article import Article
from typing import List


class Category:
    def __init__(
        self,
        id: str,
        name: str,
        child_categories: List["Category"],
        articles: list[Article],
        description: str,
        project_version_id: str,
        order: int,
        parent_category_id: str,
        hidden: bool,
        icon: str,
        slug: str,
        language_code: str,
        category_type: int,
        created_at: str,
        modified_at: str,
        status: str,
        content_type: int,
    ):
        self.id = id
        self.name = name
        self.child_categories = child_categories
        self.articles = articles
        self.description = description
        self.project_version_id = project_version_id
        self.order = order
        self.parent_category_id = parent_category_id
        self.hidden = hidden
        self.icon = icon
        self.slug = slug
        self.language_code = language_code
        self.category_type = category_type
        self.created_at = created_at
        self.modified_at = modified_at
        self.status = status
        self.content_type = content_type

    def __str__(self) -> str:
        return f"Category: {self.name} (id: {self.id})"

    def __repr__(self) -> str:
        return str(self)

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_child_categories(self):
        return self.child_categories

    def get_articles(self):
        return self.articles

    def get_description(self):
        return self.description

    def get_project_version_id(self):
        return self.project_version_id

    def get_order(self):
        return self.order

    def get_parent_category_id(self):
        return self.parent_category_id

    def get_hidden(self):
        return self.hidden

    def get_icon(self):
        return self.icon

    def get_slug(self):
        return self.slug

    def get_language_code(self):
        return self.language_code

    def get_category_type(self):
        return self.category_type

    def get_created_at(self):
        return self.created_at

    def get_modified_at(self):
        return self.modified_at

    def get_status(self):
        return self.status

    def get_content_type(self):
        return self.content_type
