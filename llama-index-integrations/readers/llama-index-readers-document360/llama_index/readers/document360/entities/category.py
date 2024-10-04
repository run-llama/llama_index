from pydantic import BaseModel
from .article_slim import ArticleSlim
from typing import List, Optional


# https://apidocs.document360.com/apidocs/project-version-categories
class Category(BaseModel):
    id: Optional[str]
    name: Optional[str]
    child_categories: Optional[List["Category"]]
    articles: Optional[List[ArticleSlim]]
    description: Optional[str]
    project_version_id: Optional[str]
    order: Optional[int]
    parent_category_id: Optional[str]
    hidden: Optional[bool]
    icon: Optional[str]
    slug: Optional[str]
    language_code: Optional[str]
    category_type: Optional[int]
    created_at: Optional[str]
    modified_at: Optional[str]
    status: Optional[int]
    content_type: Optional[int]
