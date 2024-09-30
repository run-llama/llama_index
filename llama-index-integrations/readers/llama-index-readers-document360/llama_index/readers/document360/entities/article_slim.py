# Article entity embedded into the Category entity
class ArticleSlim:
    def __init__(
        self,
        id: str,
        title: str,
        modified_at: str,
        public_version: int,
        latest_version: int,
        language_code: str,
        hidden: bool,
        status: int,
        order: int,
        slug: str,
        content_type: int,
        translation_option: int,
        is_shared_article: bool,
    ):
        self.id = id
        self.title = title
        self.modified_at = modified_at
        self.public_version = public_version
        self.latest_version = latest_version
        self.language_code = language_code
        self.hidden = hidden
        self.status = status
        self.order = order
        self.slug = slug
        self.content_type = content_type
        self.translation_option = translation_option
        self.is_shared_article = is_shared_article

    def __str__(self) -> str:
        return f"Article: {self.title} (id: {self.id})"

    def __repr__(self) -> str:
        return str(self)

    def get_id(self):
        return self.id

    def get_title(self):
        return self.title

    def get_modified_at(self):
        return self.modified_at

    def get_public_version(self):
        return self.public_version

    def get_latest_version(self):
        return self.latest_version

    def get_language_code(self):
        return self.language_code

    def get_hidden(self):
        return self.hidden

    def get_status(self):
        return self.status

    def get_order(self):
        return self.order

    def get_slug(self):
        return self.slug

    def get_content_type(self):
        return self.content_type

    def get_translation_option(self):
        return self.translation_option

    def get_is_shared_article(self):
        return self.is_shared_article
