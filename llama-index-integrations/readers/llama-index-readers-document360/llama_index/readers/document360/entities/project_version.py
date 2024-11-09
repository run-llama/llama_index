from pydantic import BaseModel
from typing import Optional, List, Dict


# https://apidocs.document360.com/apidocs/get-project-versions
class ProjectVersion(BaseModel):
    id: Optional[str]
    version_number: Optional[float]
    base_version_number: Optional[float]
    version_code_name: Optional[str]
    is_main_version: Optional[bool]
    is_beta: Optional[bool]
    is_public: Optional[bool]
    is_deprecated: Optional[bool]
    created_at: Optional[str]
    modified_at: Optional[str]
    language_versions: Optional[List[Dict]]
    slug: Optional[str]
    order: Optional[int]
    version_type: Optional[int]
