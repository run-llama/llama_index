from pydantic import BaseModel
from typing import Optional, List, Dict


# https://apidocs.document360.com/apidocs/get-article
class Article(BaseModel):
    id: Optional[str]
    title: Optional[str]
    modified_at: Optional[str]
    html_content: Optional[str]
    category_id: Optional[str]
    project_version_id: Optional[str]
    version_number: Optional[int]
    public_version: Optional[int]
    latest_version: Optional[int]
    enable_rtl: Optional[bool]
    hidden: Optional[bool]
    status: Optional[int]
    order: Optional[int]
    created_by: Optional[str]
    authors: Optional[List[Dict]]
    created_at: Optional[str]
    slug: Optional[str]
    is_fall_back_content: Optional[bool]
    description: Optional[str]
    category_type: Optional[int]
    content_type: Optional[int]
    is_shared_article: Optional[bool]
    translation_option: Optional[int]
    url: Optional[str]
    content: Optional[str]
