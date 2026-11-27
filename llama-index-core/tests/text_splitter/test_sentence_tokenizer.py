import pickle

import pytest
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.utils import globals_helper

# The danda (।) and double danda (॥) end sentences in Hindi, Marathi, Bengali and
# other languages written in Indic scripts.
SENTENCES = {
    "hindi": [
        "भारत एक विशाल देश है। ",
        "इसकी राजधानी नई दिल्ली है। ",
        "गंगा एक पवित्र नदी है।",
    ],
    "marathi": [
        "महाराष्ट्र हे भारतातील एक मोठे राज्य आहे। ",
        "मुंबई ही त्याची राजधानी आहे। ",
        "ते सर्वांसाठी धान्य पिकवतात॥",
    ],
    "bengali": [
        "বাংলা একটি সুন্দর ভাষা। ",
        "কলকাতা পশ্চিমবঙ্গের রাজধানী। ",
        "এখানে অনেক মানুষ থাকে।",
    ],
    "english": [
        "India is a large country. ",
        "Its capital is New Delhi? ",
        "Yes, and the Ganga is a holy river!",
    ],
}


@pytest.mark.parametrize("language", SENTENCES)
def test_split_by_sentence_tokenizer_ends_sentences_on_danda(language: str) -> None:
    sentences = SENTENCES[language]

    assert split_by_sentence_tokenizer()("".join(sentences)) == sentences


def test_punkt_tokenizer_keeps_danda_after_pickling() -> None:
    tokenizer = pickle.loads(pickle.dumps(globals_helper.punkt_tokenizer))
    text = "".join(SENTENCES["hindi"])

    assert [text[start:end] for start, end in tokenizer.span_tokenize(text)] == [
        sentence.rstrip() for sentence in SENTENCES["hindi"]
    ]
