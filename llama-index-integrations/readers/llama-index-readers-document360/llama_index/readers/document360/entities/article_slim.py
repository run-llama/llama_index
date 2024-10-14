from pydantic import BaseModel
from typing import Optional


# Article entity embedded into the Category entity
# https://apidocs.document360.com/apidocs/project-version-categories
class ArticleSlim(BaseModel):
    id: Optional[str]
    title: Optional[str]
    modified_at: Optional[str]
    public_version: Optional[int]
    latest_version: Optional[int]
    language_code: Optional[str]
    hidden: Optional[bool]
    status: Optional[int]
    order: Optional[int]
    slug: Optional[str]
    content_type: Optional[int]
    translation_option: Optional[int]
    is_shared_article: Optional[bool]
