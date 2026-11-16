"""Punkt language variables for the default sentence tokenizer."""

from nltk.tokenize.punkt import PunktLanguageVars


class PunktLanguageVarsWithDanda(PunktLanguageVars):
    """
    Punkt language variables that also end a sentence on the danda.

    The danda (U+0964, ``।``) and double danda (U+0965, ``॥``) end sentences in
    Hindi, Marathi, Bengali, Nepali, Sanskrit and other languages written in
    Indic scripts. Without them, Punkt treats a whole paragraph of such text as
    one sentence.

    Defined as a subclass, at module level, because ``PunktLanguageVars`` does
    not pickle instance attributes.
    """

    sent_end_chars = (*PunktLanguageVars.sent_end_chars, "।", "॥")
