"""Azure Translate tool spec."""


import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

ENDPOINT_BASE_URL = "https://api.cognitive.microsofttranslator.com/translate"


class AzureTranslateToolSpec(BaseToolSpec):
    """Azure Translate tool spec."""

    spec_functions = ["translate"]

    def __init__(self, api_key: str, region: str) -> None:
        """Initialize with parameters."""
        self.headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Ocp-Apim-Subscription-Region": region,
            "Content-type": "application/json",
        }

    def translate(self, text: str, language: str):
        """
        Use this tool to translate text from one language to another.
        The source language will be automatically detected. You need to specify the target language
        using a two character language code.
        args:
            language (str): Target translation language. One of af, sq, am, ar, hy, as, az, bn, ba, eu, bs, bg, ca, hr, cs, da, dv, nl, en, et, fo, fj, fi, fr, gl, ka, de, el, gu, ht, he, hi, hu, is, id, iu, ga, it, ja, kn, kk, km, ko, ku, ky, lo, lv, lt, mk, mg, ms, ml, mt, mi, mr, my, ne, nb, or, ps, fa, pl, pt, pa, ro, ru, sm, sk, sl, so, es, sw, sv, ty, ta, tt, te, th, bo, ti, to, tr, tk, uk, ur, ug, uz, vi, cy, zu
        """

        request = requests.post(
            ENDPOINT_BASE_URL,
            params={"api-version": "3.0", "to": language},
            headers=self.headers,
            json=[{"text": text}],
        )
        response = request.json()
        return response
