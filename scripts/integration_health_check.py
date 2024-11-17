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
from math import exp
from typing import Dict

# cache of commits to avoid re-reading from disk
commit_cache = []


class IntegrationActivityAnalyzer:
    def __init__(self, package_name: str, repo_path: str):
        self.package_name = package_name
        self.repo_path = repo_path

    def get_time_weight(self, date_str: str, decay_factor: float = 0.1) -> float:
        """Calculate time-based weight using exponential decay.

        Args:
            date_str: Date string in YYYY-MM-DD format
            decay_factor: Controls how quickly the weight decays (higher = faster decay)

        Returns:
            Weight between 0 and 1, with more recent dates closer to 1
        """
        date = datetime.strptime(date_str, "%Y-%m-%d")
        days_ago = (datetime.now() - date).days
        return exp(-decay_factor * days_ago / 30)  # Normalize by month

    def get_download_trends(self) -> Dict[str, float]:
        """Get download trends for the package.

        Analyzes PyPI download trends over the last 12 months
        Returns trend metrics including growth rate and stability
        """
        # Using PyPI Stats API for monthly data
        try:
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
        except Exception as e:
            return {
                "growth_rate": 0,
                "stability": 0,
                "avg_monthly_downloads": 0,
            }

        downloads_per_month = {}
        for item in response["data"]:
            if item["date"] not in downloads_per_month:
                downloads_per_month[item["date"]] = item["downloads"]
            else:
                downloads_per_month[item["date"]] += item["downloads"]

        # We need at least 5 months of data, if not, its too new to be considered
        if len(downloads_per_month) < 5:
            return None

        # Apply time weights to downloads
        weighted_downloads = []
        for date, downloads in downloads_per_month.items():
            weight = self.get_time_weight(date + "-01")
            weighted_downloads.append(downloads * weight)

        # Calculate growth rates with weighted values
        growth_rates = []
        for i in range(1, len(weighted_downloads)):
            if weighted_downloads[i - 1] == 0:
                continue
            growth_rate = (
                weighted_downloads[i] - weighted_downloads[i - 1]
            ) / weighted_downloads[i - 1]
            growth_rates.append(growth_rate)

        avg_growth_rate = statistics.mean(growth_rates) if growth_rates else 0
        stability = statistics.stdev(growth_rates) if len(growth_rates) > 1 else 1

        return {
            "growth_rate": avg_growth_rate,
            "stability": stability,
            "avg_monthly_downloads": statistics.mean(weighted_downloads),
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

        # use cache if available
        if len(commit_cache) == 0:
            for commit in repo.iter_commits(since=six_months_ago):
                # Check if any files in this commit are in the package directory
                for file in commit.stats.files:
                    if self.package_name in file:
                        commits.append(commit)
                commit_cache.append(commit)
        else:
            for commit in commit_cache:
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

        # Apply time weights to commit counts
        weighted_monthly_commits = {}
        for month_key, commit_count in monthly_commits.items():
            # Convert month_key to date string (use first day of month)
            date_str = f"{month_key}-01"
            weight = self.get_time_weight(date_str)
            weighted_monthly_commits[month_key] = commit_count * weight

        commit_counts = list(weighted_monthly_commits.values())
        avg_monthly_commits = statistics.mean(commit_counts) if commit_counts else 0
        commit_consistency = (
            statistics.stdev(commit_counts) if len(commit_counts) > 1 else 1
        )

        return {
            "commit_frequency": avg_monthly_commits,
            "commit_consistency": commit_consistency,
            "total_commits": len(commits),  # Could also weight this but less meaningful
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

        # if the package is too new to have any data, return an arbitrary high score
        if current_metrics["downloads"] is None or current_metrics["commits"] is None:
            return 1000

        # Calculate ratios relative to core package (current/core)
        download_ratio = (
            current_metrics["downloads"]["avg_monthly_downloads"]
            / core_package_metrics["downloads"]["avg_monthly_downloads"]
        )

        download_stability_ratio = (
            current_metrics["downloads"]["stability"]
            / core_package_metrics["downloads"]["stability"]
        )

        download_growth_ratio = (
            current_metrics["downloads"]["growth_rate"]
            / core_package_metrics["downloads"]["growth_rate"]
        )

        commit_ratio = (
            current_metrics["commits"]["total_commits"]
            / core_package_metrics["commits"]["total_commits"]
        )

        commit_consistency_ratio = (
            current_metrics["commits"]["commit_consistency"]
            / core_package_metrics["commits"]["commit_consistency"]
        )

        commit_frequency_ratio = (
            current_metrics["commits"]["commit_frequency"]
            / core_package_metrics["commits"]["commit_frequency"]
        )

        # Weight the different factors
        # Max score is 1.0
        ratios = [download_ratio * 2.0, commit_ratio * 0.5]
        return sum(ratios) / len(ratios)


def analyze_multiple_packages(
    package_paths: list[str], bottom_percent: float = 0.25
) -> list[tuple[str, float]]:
    """
    Analyze multiple packages and return the bottom X% scoring packages.

    Args:
        package_paths: List of package paths to analyze
        bottom_percent: Percentage of lowest scoring packages to return (0.0 to 1.0)

    Returns:
        List of tuples containing (package_name, health_score) sorted by score ascending
    """
    if os.path.exists("./all_package_metrics.json"):
        with open("./all_package_metrics.json") as f:
            results = json.load(f)
    else:
        results = []
        for package_path in package_paths:
            package_name = package_path.strip().lstrip("./").rstrip("/").split("/")[-1]
            analyzer = IntegrationActivityAnalyzer(package_name, package_path)
            health_score = analyzer.calculate_relative_health()
            results.append((package_name, health_score))

        with open("./all_package_metrics.json", "w") as f:
            json.dump(results, f)

    # Sort by health score ascending
    results.sort(key=lambda x: x[1])

    # Calculate how many packages to return
    num_packages = max(1, int(len(results) * bottom_percent))

    return results[:num_packages]


if __name__ == "__main__":
    arg = sys.argv[1]
    try:
        percent = float(arg)
        all_packages = []
        for root, dirs, files in os.walk("./llama-index-integrations"):
            if "pyproject.toml" in files:
                all_packages.append(root)

        packages_to_remove = analyze_multiple_packages(all_packages, percent)
        print(f"Found {len(packages_to_remove)} packages to remove.")
        print("\n".join([str(x) for x in packages_to_remove]))
    except ValueError:
        package_path = sys.argv[1].strip().lstrip("./").rstrip("/")
        package_name = package_path.split("/")[-1]
        analyzer = IntegrationActivityAnalyzer(package_name, package_path)
        print(analyzer.calculate_relative_health())
