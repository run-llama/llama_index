"""General utils functions."""

from typing import List, Optional

import nltk
from transformers import GPT2TokenizerFast


class GlobalsHelper:
    """Helper to retrieve globals.

    Helpful for global caching of certain variables that can be expensive to load.
    (e.g. tokenization)

    """

    _tokenizer: Optional[GPT2TokenizerFast] = None
    _stopwords: Optional[List[str]] = None

    @property
    def tokenizer(self) -> GPT2TokenizerFast:
        """Get tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        return self._tokenizer

    @property
    def stopwords(self) -> List[str]:
        """Get stopwords."""
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
            except ImportError:
                raise ValueError(
                    "`nltk` package not found, please run `pip install nltk`"
                )
            nltk.download("stopwords")
            self._stopwords = stopwords.words("english")
        return self._stopwords


globals_helper = GlobalsHelper()
