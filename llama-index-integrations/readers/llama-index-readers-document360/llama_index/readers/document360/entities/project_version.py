class ProjectVersion:
    def __init__(
        self,
        id: str,
        version_number: float,
        base_version_number: float,
        version_code_name: str,
        is_main_version: bool,
        is_beta: bool,
        is_public: bool,
        is_deprecated: bool,
        created_at: str,
        modified_at: str,
        language_versions: list[dict],
        slug: str,
        order: int,
        version_type: int,
    ):
        self.id = id
        self.version_number = version_number
        self.base_version_number = base_version_number
        self.version_code_name = version_code_name
        self.is_main_version = is_main_version
        self.is_beta = is_beta
        self.is_public = is_public
        self.is_deprecated = is_deprecated
        self.created_at = created_at
        self.modified_at = modified_at
        self.language_versions = language_versions
        self.slug = slug
        self.order = order
        self.version_type = version_type

    def __str__(self) -> str:
        return f"ProjectVersion: {self.version_code_name} (id: {self.id})"

    def __repr__(self) -> str:
        return str(self)

    def get_id(self):
        return self.id

    def get_version_number(self):
        return self.version_number

    def get_base_version_number(self):
        return self.base_version_number

    def get_version_code_name(self):
        return self.version_code_name

    def get_is_main_version(self):
        return self.is_main_version

    def get_is_beta(self):
        return self.is_beta

    def get_is_public(self):
        return self.is_public

    def get_is_deprecated(self):
        return self.is_deprecated

    def get_created_at(self):
        return self.created_at

    def get_modified_at(self):
        return self.modified_at

    def get_language_versions(self):
        return self.language_versions

    def get_slug(self):
        return self.slug

    def get_order(self):
        return self.order

    def get_version_type(self):
        return self.version_type
