class Article:
    def __init__(
        self,
        id: str,
        title: str,
        modified_at: str,
        html_content: str,
        category_id: str,
        project_version_id: str,
        version_number: int,
        public_version: int,
        latest_version: int,
        enable_rtl: bool,
        hidden: bool,
        status: str,
        order: int,
        created_by: str,
        authors: list[dict],
        created_at: str,
        slug: str,
        is_fall_back_content: bool,
        description: str,
        category_type: int,
        content_type: int,
        is_shared_article: bool,
        translation_option: int,
        url: str,
        content: str,
    ):
        self.id = id
        self.title = title
        self.content = content
        self.html_content = html_content
        self.category_id = category_id
        self.project_version_id = project_version_id
        self.version_number = version_number
        self.public_version = public_version
        self.latest_version = latest_version
        self.enable_rtl = enable_rtl
        self.hidden = hidden
        self.status = status
        self.order = order
        self.created_by = created_by
        self.authors = authors
        self.created_at = created_at
        self.modified_at = modified_at
        self.slug = slug
        self.is_fall_back_content = is_fall_back_content
        self.description = description
        self.category_type = category_type
        self.content_type = content_type
        self.is_shared_article = is_shared_article
        self.translation_option = translation_option
        self.url = url

    def __str__(self) -> str:
        return f"Article: {self.title} (id: {self.id})"

    def __repr__(self) -> str:
        return str(self)

    def get_id(self):
        return self.id

    def get_title(self):
        return self.title

    def get_content(self):
        return self.content

    def get_html_content(self):
        return self.html_content

    def get_category_id(self):
        return self.category_id

    def get_project_version_id(self):
        return self.project_version_id

    def get_version_number(self):
        return self.version_number

    def get_public_version(self):
        return self.public_version

    def get_latest_version(self):
        return self.latest_version

    def get_enable_rtl(self):
        return self.enable_rtl

    def get_hidden(self):
        return self.hidden

    def get_status(self):
        return self.status

    def get_order(self):
        return self.order

    def get_created_by(self):
        return self.created_by

    def get_authors(self):
        return self.authors

    def get_created_at(self):
        return self.created_at

    def get_modified_at(self):
        return self.modified_at

    def get_slug(self):
        return self.slug

    def get_is_fall_back_content(self):
        return self.is_fall_back_content

    def get_description(self):
        return self.description

    def get_category_type(self):
        return self.category_type

    def get_content_type(self):
        return self.content_type

    def get_is_shared_article(self):
        return self.is_shared_article

    def get_translation_option(self):
        return self.translation_option

    def get_url(self):
        return self.url
