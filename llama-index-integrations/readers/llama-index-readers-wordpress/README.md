# Wordpress Loader

```bash
pip install llama-index-readers-wordpress
```

This loader fetches the text from Wordpress blog posts using the Wordpress API. It also uses the BeautifulSoup library to parse the HTML and extract the text from the articles.

## Usage

To use this loader, you need to pass base url of the Wordpress installation
(e.g. `https://www.mysite.com`) and optionally a username, and an application
password for the user (more about application passwords
[here](https://www.paidmembershipspro.com/create-application-password-wordpress/))

```python
from llama_index.readers.wordpress import WordpressReader

loader = WordpressReader(
    url="https://www.mysite.com",
    username="my_username",
    password="my_password",
)
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/).

## Pages and Posts

Be default, the loader retrieves both Wordpress _pages_ (static content) and
_posts_ (blog entries) from the target site. This behavior can be configured
by setting `get_pages=False` or `get_posts=False` when initializing the
`WordpressReader` object.
