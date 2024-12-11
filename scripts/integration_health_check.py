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
import concurrent.futures
from functools import lru_cache
import pathlib
import ast
import json
from typing import Dict, List
import pandas as pd

from typing import Literal
from datetime import datetime, timedelta
from math import exp

# cache of commits to avoid re-reading from disk
commit_cache = []

DEFAULT_METRIC_WEIGHTS = {
    "download_ratio": 0.40,
    "download_stability_ratio": 0.0,
    "download_growth_ratio": 0.0,
    "commit_ratio": 0.10,
    "commit_consistency_ratio": 0.0,
    "commit_frequency_ratio": 0.0,
    "test_score": 0.50,
}

DEFAULT_SCORE_NEW_PROJECT = 1.0


class IntegrationActivityAnalyzer:
    def __init__(
        self,
        package_name: str,
        repo_path: str,
        metric_weights: Dict = DEFAULT_METRIC_WEIGHTS,
        new_project_score: float = DEFAULT_SCORE_NEW_PROJECT,
        verbose: bool = False,
    ):
        self.package_name = package_name
        self.repo_path = repo_path
        self.metrics = {}
        self.verbose = verbose
        if sum(v for v in metric_weights.values()) != 1:
            raise ValueError("Metric weights do not sum up to 1.")
        self.metric_weights = metric_weights
        self._is_new_project = False
        self._new_project_score = new_project_score

    def get_time_weight(self, date_str: str, decay_factor: float = 0.5) -> float:
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

    @lru_cache(maxsize=128)
    def get_download_trends(self) -> Dict[str, float]:
        """Get download trends for the package.
        Cache results to avoid repeated PyPI API calls.
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
            self._is_new_project = True
            return {
                "growth_rate": 0,
                "stability": 0,
                "avg_monthly_downloads": 0,
            }

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
        """Get commit activity for the package."""
        repo = git.Repo("./")
        now = datetime.now()
        six_months_ago = now - timedelta(days=365 // 2)

        # Get all commits once and cache them at module level
        global commit_cache
        if not commit_cache:
            # Use a set for faster lookups
            commit_cache = {
                commit: {file for file in commit.stats.files}  # noqa: C416
                for commit in repo.iter_commits(since=six_months_ago)
            }

        # Filter commits for this package more efficiently
        commits = [
            commit
            for commit, files in commit_cache.items()
            if any(self.package_name in file for file in files)
        ]

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

    def _count_tests_in_file(self, file_path: pathlib.Path) -> int:
        """
        Count the number of test functions in a Python file.
        Looks for functions that start with 'test_' or methods in classes that start with 'Test'.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read())

            test_count = 0

            for node in ast.walk(tree):
                # Count standalone test functions
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_count += 1

                # Count test methods in test classes
                elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    test_methods = [
                        method
                        for method in node.body
                        if isinstance(method, ast.FunctionDef)
                        and (method.name.startswith("test_") or method.name == "test")
                    ]
                    test_count += len(test_methods)

            return test_count
        except Exception:
            # If we can't parse the file, return 0
            return 0

    def check_test_coverage(self) -> float:
        """
        Check if package has adequate test coverage.
        Returns 1.0 if package has at least 5 test functions, 0.5 if it has 2-4 tests,
        and 0.0 if it has less than 2 tests.
        """
        package_path = pathlib.Path(self.repo_path)

        # Look for tests in common test directory locations
        test_files: List[pathlib.Path] = []
        test_dirs = [
            package_path / "tests",
            package_path / "test",
            package_path.parent / "tests" / package_path.name,
        ]

        for test_dir in test_dirs:
            if test_dir.exists() and test_dir.is_dir():
                test_files.extend(test_dir.glob("test_*.py"))
                test_files.extend(test_dir.glob("*_test.py"))

        # Count total number of test functions across all files
        total_tests = sum(self._count_tests_in_file(file) for file in test_files)

        # Return score based on number of tests
        if total_tests >= 5:
            return 1.0
        elif total_tests >= 2:
            return 0.5
        else:
            return 0.0

    def calculate_metrics(self) -> None:
        """
        Calculate relative health score compared to llama-index-core.
        """
        if os.path.exists("./core_package_metrics.json"):
            if self.verbose:
                print(
                    "Loading cached existing core package metrics from ./core_package_metrics.json"
                )
            with open("./core_package_metrics.json") as f:
                core_package_metrics = json.load(f)
        else:
            if self.verbose:
                print("No cached existing core package metrics found, calculating...")
            core_package_metrics = {
                "downloads": IntegrationActivityAnalyzer(
                    repo_path="./llama-index-core", package_name="llama-index-core"
                ).get_download_trends(),
                "commits": IntegrationActivityAnalyzer(
                    repo_path="./llama-index-core", package_name="llama-index-core"
                ).get_commit_activity(),
            }
            with open("./core_package_metrics.json", "w") as f:
                json.dump(core_package_metrics, f)

        current_metrics = {
            "downloads": self.get_download_trends(),
            "commits": self.get_commit_activity(),
        }

        # if the package is too new to have any data, set new project flag
        if current_metrics["downloads"] is None or current_metrics["commits"] is None:
            self._is_new_project = True

        # Calculate ratios relative to core package (current/core)
        self.metrics["download_ratio"] = (
            current_metrics["downloads"]["avg_monthly_downloads"]
            / core_package_metrics["downloads"]["avg_monthly_downloads"]
        )

        self.metrics["download_stability_ratio"] = (
            current_metrics["downloads"]["stability"]
            / core_package_metrics["downloads"]["stability"]
        )

        self.metrics["download_growth_ratio"] = (
            current_metrics["downloads"]["growth_rate"]
            / core_package_metrics["downloads"]["growth_rate"]
        )

        self.metrics["commit_ratio"] = (
            current_metrics["commits"]["total_commits"]
            / core_package_metrics["commits"]["total_commits"]
        )

        self.metrics["commit_consistency_ratio"] = (
            current_metrics["commits"]["commit_consistency"]
            / core_package_metrics["commits"]["commit_consistency"]
        )

        self.metrics["commit_frequency_ratio"] = (
            current_metrics["commits"]["commit_frequency"]
            / core_package_metrics["commits"]["commit_frequency"]
        )

        # Weight the different factors
        # Max score is 1.0
        self.metrics["test_score"] = self.check_test_coverage()

    @property
    def health_score(self) -> float:
        if self._is_new_project:
            score = self._new_project_score
        else:
            score = 0
            for k, v in self.metrics.items():
                score += v * self.metric_weights[k]
        return score


def analyze_package(package_path: str) -> tuple[str, float]:
    """Analyze a single package. Helper function for parallel processing."""
    package_name = package_path.strip().lstrip("./").rstrip("/").split("/")[-1]
    logger.info(f"starting to analyze {package_name}")
    analyzer = IntegrationActivityAnalyzer(package_name, package_path)
    analyzer.calculate_metrics()
    health_score = analyzer.health_score
    if health_score == 42:
        print(f"new package: {package_name}")
    logger.info(f"health score for {package_name}: {health_score}")
    return (package_name, health_score)


def analyze_multiple_packages(
    package_paths: list[str],
    bottom_percent: float | None = None,
    bottom_n: int | None = None,
    threshold: float | None = None,
) -> list[tuple[str, float]]:
    """Analyze multiple packages in parallel."""
    if os.path.exists("./all_package_metrics.json"):
        print("Loading cached existing package metrics from ./all_package_metrics.json")
        with open("./all_package_metrics.json") as f:
            results = json.load(f)
    else:
        print("No cached existing package metrics found, calculating...")
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(analyze_package, package_paths))

        with open("./all_package_metrics.json", "w") as f:
            json.dump(results, f)

    # Sort by health score ascending
    results.sort(key=lambda x: x[1])

    # Print summary stats
    scores = pd.Series([el[1] for el in results])
    print(scores.describe())

    # Calculate how many packages to return
    if bottom_percent is not None:
        num_packages = max(1, int(len(results) * bottom_percent))
    elif bottom_n is not None:
        num_packages = min(bottom_n, len(results))
    elif threshold is not None:
        num_packages = next(
            (i for i, (_, score) in enumerate(results) if score >= threshold),
            len(results),
        )
    else:
        raise ValueError(
            "Either bottom_percent or threshold must be provided, but not both."
        )

    return results[:num_packages]


def package_tuple_to_str(
    package_tuple: tuple[str, float], mode: Literal["default", "csv"] = "default"
):
    if mode == "default":
        return str(package_tuple)
    elif mode == "csv":
        name, score = package_tuple
        return f"{name},{score}"
    else:
        raise ValueError(
            "Unsupported str mode. Please enter `default` or `csv` as mode."
        )


if __name__ == "__main__":
    arg = sys.argv[1]
    try:
        val = float(arg)
        is_threshold = sys.argv[2] == "threshold"
        is_percent = sys.argv[2] == "percent"
        is_bottom_n = sys.argv[2] = "bottom_n"
        try:
            output_mode = sys.argv[3]
        except IndexError:
            output_mode = "default"
        all_packages = []
        for root, dirs, files in os.walk("./llama-index-integrations"):
            if "pyproject.toml" in files:
                all_packages.append(root)
        for root, dirs, files in os.walk("./llama-index-packs"):
            if "pyproject.toml" in files:
                all_packages.append(root)

        if is_percent:
            packages_to_remove = analyze_multiple_packages(
                all_packages, bottom_percent=val
            )
        elif is_bottom_n:
            packages_to_remove = analyze_multiple_packages(
                all_packages, bottom_n=int(val)
            )
        elif is_threshold:
            packages_to_remove = analyze_multiple_packages(all_packages, threshold=val)
        else:
            raise ValueError("Invalid argument for bottom_percent or threshold")

        print(f"Found {len(packages_to_remove)} packages to remove.")
        print(
            "\n".join(
                [package_tuple_to_str(x, mode=output_mode) for x in packages_to_remove]
            )
        )

    except ValueError:
        package_path = sys.argv[1].strip().lstrip("./").rstrip("/")
        package_name = package_path.split("/")[-1]
        print(f"{package_name} at {package_path}")
        analyzer = IntegrationActivityAnalyzer(package_name, package_path)
        analyzer.calculate_metrics()
        print("metrics dict:\n", json.dumps(analyzer.metrics, indent=4))
        print("health score:\n", analyzer.health_score)
