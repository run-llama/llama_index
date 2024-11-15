"""
Calculate the relative health of a package compared to llama-index-core.

Output is a score between 0 and 1.

At the time of writing, llama-index-llms-openai has a score of 0.38.

Example usage:
python ./integration_health_check.py llama-index-integrations/llms/llama-index-llms-openai
"""

import git
import json
import os
import pypistats
import statistics
import sys

from datetime import datetime, timedelta
from typing import Dict


class IntegrationActivityAnalyzer:
    def __init__(self, package_name: str, repo_path: str):
        self.package_name = package_name
        self.repo_path = repo_path

    def get_download_trends(self) -> Dict[str, float]:
        """Get download trends for the package.

        Analyzes PyPI download trends over the last 12 months
        Returns trend metrics including growth rate and stability
        """
        # Using PyPI Stats API for monthly data
        todays_date = datetime.now().strftime("%Y-%m-%d")
        one_eighty_days_ago = (datetime.now() - timedelta(days=180)).strftime(
            "%Y-%m-%d"
        )
        response_str = pypistats.overall(
            self.package_name,
            format="json",
            total="monthly",
            start_date=one_eighty_days_ago,
            end_date=todays_date,
        )
        response = json.loads(response_str)

        downloads_per_month = {}
        for item in response["data"]:
            if item["date"] not in downloads_per_month:
                downloads_per_month[item["date"]] = item["downloads"]
            else:
                downloads_per_month[item["date"]] += item["downloads"]

        downloads = list(downloads_per_month.values())

        # Calculate month-over-month growth rate
        growth_rates = []
        for i in range(1, len(downloads)):
            if downloads[i - 1] == 0:
                continue
            growth_rate = (downloads[i] - downloads[i - 1]) / downloads[i - 1]
            growth_rates.append(growth_rate)

        avg_growth_rate = statistics.mean(growth_rates) if growth_rates else 0
        stability = statistics.stdev(growth_rates) if len(growth_rates) > 1 else 1

        return {
            "growth_rate": avg_growth_rate,
            "stability": stability,
            "avg_monthly_downloads": statistics.mean(downloads),
        }

    def get_commit_activity(self) -> Dict[str, float]:
        """Get commit activity for the package.

        Analyzes git commit patterns over the last 6 months
        Returns commit frequency and consistency metrics
        """
        repo = git.Repo("./")
        now = datetime.now()
        six_months_ago = now - timedelta(days=365 // 2)

        # Get commits from the last year that modified files in the package directory
        commits = []
        for commit in repo.iter_commits(since=six_months_ago):
            # Check if any files in this commit are in the package directory
            for file in commit.stats.files:
                if self.package_name in file:
                    commits.append(commit)
                    break

        if not commits:
            return {"commit_frequency": 0, "commit_consistency": 0, "total_commits": 0}

        # Rest of the method remains the same
        monthly_commits = {}
        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            month_key = f"{commit_date.year}-{commit_date.month}"
            monthly_commits[month_key] = monthly_commits.get(month_key, 0) + 1

        # Calculate commit frequency and consistency
        commit_counts = list(monthly_commits.values())
        avg_monthly_commits = statistics.mean(commit_counts) if commit_counts else 0
        commit_consistency = (
            statistics.stdev(commit_counts) if len(commit_counts) > 1 else 1
        )

        return {
            "commit_frequency": avg_monthly_commits,
            "commit_consistency": commit_consistency,
            "total_commits": len(commits),
        }

    def calculate_relative_health(self) -> float:
        """
        Calculate relative health score compared to llama-index-core.
        """
        if os.path.exists("./core_package_metrics.json"):
            with open("./core_package_metrics.json") as f:
                core_package_metrics = json.load(f)
        else:
            core_package_metrics = {
                "downloads": IntegrationActivityAnalyzer(
                    "./llama-index-core", "llama-index-core"
                ).get_download_trends(),
                "commits": IntegrationActivityAnalyzer(
                    "./llama-index-core", "llama-index-core"
                ).get_commit_activity(),
            }
            with open("./core_package_metrics.json", "w") as f:
                json.dump(core_package_metrics, f)

        current_metrics = {
            "downloads": self.get_download_trends(),
            "commits": self.get_commit_activity(),
        }

        # Calculate ratios relative to core package (current/core)
        download_ratio = min(
            1.0,
            current_metrics["downloads"]["avg_monthly_downloads"]
            / core_package_metrics["downloads"]["avg_monthly_downloads"],
        )

        download_stability_ratio = min(
            1.0,
            current_metrics["downloads"]["stability"]
            / core_package_metrics["downloads"]["stability"],
        )

        download_growth_ratio = min(
            1.0,
            current_metrics["downloads"]["growth_rate"]
            / core_package_metrics["downloads"]["growth_rate"],
        )

        commit_ratio = min(
            1.0,
            current_metrics["commits"]["total_commits"]
            / core_package_metrics["commits"]["total_commits"],
        )

        commit_consistency_ratio = min(
            1.0,
            current_metrics["commits"]["commit_consistency"]
            / core_package_metrics["commits"]["commit_consistency"],
        )

        commit_frequency_ratio = min(
            1.0,
            current_metrics["commits"]["commit_frequency"]
            / core_package_metrics["commits"]["commit_frequency"],
        )

        # Weight the different factors
        # Max score is 1.0
        ratios = [download_ratio, commit_ratio]
        return sum(ratios) / len(ratios)


if __name__ == "__main__":
    package_path = sys.argv[1].strip().lstrip("./").rstrip("/")
    package_name = package_path.split("/")[-1]
    analyzer = IntegrationActivityAnalyzer(package_name, package_path)
    print(analyzer.calculate_relative_health())
