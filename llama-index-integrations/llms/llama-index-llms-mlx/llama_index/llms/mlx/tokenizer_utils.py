import json
from functools import partial

from transformers import AutoTokenizer

REPLACEMENT_CHAR = "\ufffd"


def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError

    def add_token(self, token):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        self.offset = 0
        self._tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)

    def finalize(self):
        self._tokens.extend(self._current_tokens)
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text[-1] == "\n":
            self._tokens.extend(self._current_tokens)
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def tokens(self):
        return self._tokens


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer, trim_space=True) -> None:
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        # Replace bytes with their value
        for i in range(len(self.tokenmap)):
            if self.tokenmap[i].startswith("<0x"):
                self.tokenmap[i] = chr(int(self.tokenmap[i][3:5], 16))

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        if v[0] == "\u2581":
            if self.text or not self.trim_space:
                self.text += self._unflushed.replace("\u2581", " ")
            else:
                self.text = _remove_space(self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        if self.text or not self.trim_space:
            self.text += self._unflushed.replace("\u2581", " ")
        else:
            self.text = _remove_space(self._unflushed.replace("\u2581", " "))
        self._unflushed = ""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None

    def __init__(self, tokenizer, trim_space=False) -> None:
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        # if the token starts with space
        if self._byte_decoder[v[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8"
        )
        if self.text or not self.trim_space:
            self.text += current_text
        else:
            self.text += _remove_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer) -> None:
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

    def __getattr__(self, attr) -> object:
        if attr == "detokenizer":
            return self._detokenizer
        else:
            return getattr(self._tokenizer, attr)


def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    _target_description = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }

    return _match(_target_description, decoder)


def load_tokenizer(model_path, tokenizer_config_extra={}):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        tokenizer_content = json.load(tokenizer_file.open())
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    return TokenizerWrapper(
        AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
        detokenizer_class,
    )
