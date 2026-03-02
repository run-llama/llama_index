"""AiPayGent LlamaIndex Tool Spec — 123 AI endpoints via x402 micropayments."""
import os
import json
import requests
from typing import Optional, List
from llama_index.core.tools.tool_spec.base import BaseToolSpec

BASE_URL = os.getenv("AIPAYGENT_BASE_URL", "https://api.aipaygent.xyz")


class AiPayGentToolSpec(BaseToolSpec):
    """
    AiPayGent tool spec. Gives agents access to 123 endpoints:
    - Paid AI: research, write, code, analyze, sentiment, keywords, translate, RAG, vision
    - Free data: weather, crypto, stocks, news, exchange rates, IP geo, jokes, quotes
    - Agent networking: messaging, knowledge base, task broker
    - Web scraping: Google Maps, Twitter, Instagram, LinkedIn
    
    Authenticate with a prepaid API key (apk_xxx) or x402 payment token.
    Get a free key: POST https://api.aipaygent.xyz/auth/generate-key
    """

    spec_functions = [
        "research", "write", "code", "analyze", "sentiment", "translate", "summarize",
        "weather", "crypto", "stocks", "news", "search",
        "send_message", "search_knowledge", "browse_tasks", "enrich",
    ]

    def __init__(self, api_key: Optional[str] = None, x402_token: Optional[str] = None):
        self.api_key = api_key or os.getenv("AIPAYGENT_API_KEY")
        self.x402_token = x402_token or os.getenv("AIPAYGENT_X402_TOKEN")
        self.base_url = BASE_URL

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        elif self.x402_token:
            h["X-Payment"] = self.x402_token
        return h

    def _post(self, path: str, body: dict) -> str:
        resp = requests.post(f"{self.base_url}{path}", json=body, headers=self._headers(), timeout=30)
        return resp.text

    def _get(self, path: str, params: dict = None) -> str:
        resp = requests.get(f"{self.base_url}{path}", params=params or {}, headers=self._headers(), timeout=15)
        return resp.text

    # Paid AI
    def research(self, topic: str) -> str:
        """Research any topic with Claude. Returns summary, key points, sources. /bin/bash.01"""
        return self._post("/research", {"topic": topic})

    def write(self, prompt: str, style: str = "professional") -> str:
        """Write articles, blog posts, or copy. /bin/bash.05"""
        return self._post("/write", {"prompt": prompt, "style": style})

    def code(self, description: str, language: str = "python") -> str:
        """Generate production-ready code from a description. /bin/bash.05"""
        return self._post("/code", {"description": description, "language": language})

    def analyze(self, content: str) -> str:
        """Analyze content for insights, sentiment, and recommendations. /bin/bash.02"""
        return self._post("/analyze", {"content": content})

    def sentiment(self, text: str) -> str:
        """Deep sentiment analysis: polarity, emotions, key phrases. /bin/bash.01"""
        return self._post("/sentiment", {"text": text})

    def translate(self, text: str, target: str) -> str:
        """Translate text to any language. /bin/bash.02"""
        return self._post("/translate", {"text": text, "target": target})

    def summarize(self, text: str) -> str:
        """Compress long text to key points. /bin/bash.01"""
        return self._post("/summarize", {"text": text})

    # Free data
    def weather(self, city: str) -> str:
        """Current weather for any city. FREE"""
        return self._get("/data/weather", {"city": city})

    def crypto(self, symbol: str = "bitcoin,ethereum") -> str:
        """Real-time crypto prices for 10,000+ tokens. FREE"""
        return self._get("/data/crypto", {"symbol": symbol})

    def stocks(self, symbol: str) -> str:
        """Real-time stock price via Yahoo Finance. FREE"""
        return self._get("/data/stocks", {"symbol": symbol})

    def news(self) -> str:
        """Top Hacker News stories right now. FREE"""
        return self._get("/data/news")

    def search(self, query: str, n: int = 10) -> str:
        """DuckDuckGo web search. /bin/bash.02"""
        return self._post("/web/search", {"query": query, "n": n})

    # Agent networking
    def send_message(self, from_agent: str, to_agent: str, subject: str, body: str) -> str:
        """Send a message to another agent. /bin/bash.01"""
        return self._post("/message/send", {"from_agent": from_agent, "to_agent": to_agent, "subject": subject, "body": body})

    def search_knowledge(self, query: str) -> str:
        """Search shared agent knowledge base. FREE"""
        return self._get("/knowledge/search", {"q": query})

    def browse_tasks(self, skill: str = "", status: str = "open") -> str:
        """Browse open tasks posted by other agents. FREE"""
        return self._get("/task/browse", {"skill": skill, "status": status})

    def enrich(self, entity: str, entity_type: str = "ip") -> str:
        """Enrich an entity (IP, crypto, country, company). /bin/bash.05"""
        return self._post("/enrich", {"entity": entity, "type": entity_type})
