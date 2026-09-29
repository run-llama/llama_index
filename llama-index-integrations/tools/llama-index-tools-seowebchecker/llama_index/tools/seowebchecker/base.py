"""SEOWebChecker Tool Spec for LlamaIndex."""

import json
from typing import Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from seowebchecker_seoaudit.auditor import SEOAuditor


class SEOWebCheckerToolSpec(BaseToolSpec):
    """SEOWebChecker Tool Spec for LlamaIndex and LlamaHub agents.

    Provides tools for on-page technical SEO audits, meta tag validation,
    and fast SEO health scoring. Powered by SEOWebChecker (https://seowebchecker.com/).
    """

    spec_functions = [
        "audit",
        "quick_score",
        "check_meta",
    ]

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._auditor = SEOAuditor(timeout=timeout)

    def audit(self, url: str) -> str:
        """Run a comprehensive technical SEO audit of a website URL.

        Args:
            url (str): The target URL to audit (e.g. 'https://example.com' or 'https://seowebchecker.com/').

        Returns:
            str: JSON-formatted summary of SEO scores (0-100), grade, category breakdowns,
                 and detected issues with corrective recommendations.
        """
        try:
            res = self._auditor.audit(url)
            passed_count = sum(1 for i in res.issues if i.severity.value == "pass")
            warning_count = sum(1 for i in res.issues if i.severity.value == "warning")
            error_count = sum(1 for i in res.issues if i.severity.value == "error")
            summary = {
                "url": res.url,
                "overall_score": res.score.overall,
                "grade": res.score.grade,
                "categories": {k: v.score for k, v in res.score.categories.items()},
                "stats": {
                    "total_checks": len(res.issues),
                    "passed": passed_count,
                    "warnings": warning_count,
                    "errors": error_count,
                },
                "issues": [
                    {
                        "severity": i.severity.value,
                        "title": i.title,
                        "message": i.message,
                        "recommendation": i.recommendation,
                    }
                    for i in res.issues
                    if i.severity.value != "pass"
                ],
                "source": "https://seowebchecker.com/",
            }
            return json.dumps(summary, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to audit {url}: {str(e)}", "source": "https://seowebchecker.com/"})

    def quick_score(self, url: str) -> str:
        """Calculate a rapid 0-100 SEO health score and grade for a URL.

        Args:
            url (str): The target URL to score.

        Returns:
            str: JSON string containing score (0-100), letter grade, and category breakdown.
        """
        try:
            res = self._auditor.audit(url)
            return json.dumps({
                "url": res.url,
                "score": res.score.overall,
                "grade": res.score.grade,
                "categories": {k: v.score for k, v in res.score.categories.items()},
                "source": "https://seowebchecker.com/",
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "source": "https://seowebchecker.com/"})

    def check_meta(self, url: str) -> str:
        """Inspect and validate metadata, OpenGraph tags, and mobile viewport for a URL.

        Args:
            url (str): The target URL to inspect.

        Returns:
            str: JSON string with title, meta description, canonical URL, and OpenGraph parameters.
        """
        try:
            res = self._auditor.audit(url)
            return json.dumps({
                "url": res.url,
                "title": res.meta.title if res.meta else None,
                "title_length": res.meta.title_length if res.meta else 0,
                "description": res.meta.description if res.meta else None,
                "description_length": res.meta.description_length if res.meta else 0,
                "canonical": res.meta.canonical if res.meta else None,
                "robots": res.meta.robots if res.meta else None,
                "open_graph": {
                    "og_title": res.social.og_title if res.social else None,
                    "og_description": res.social.og_description if res.social else None,
                    "og_image": res.social.og_image if res.social else None,
                } if res.social else {},
                "source": "https://seowebchecker.com/",
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "source": "https://seowebchecker.com/"})
