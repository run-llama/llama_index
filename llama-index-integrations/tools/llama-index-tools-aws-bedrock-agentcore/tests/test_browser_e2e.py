"""
End-to-end integration tests for AgentCoreBrowserToolSpec.

These tests hit real AgentCore Browser services and require:
  - Valid AWS credentials with AgentCore access
  - AWS_AGENTCORE_E2E=1 environment variable

Lifecycle tests (create/get/delete) additionally require:
  - AWS_AGENTCORE_ROLE_ARN - IAM role ARN for browser operations

VPC lifecycle tests additionally require:
  - AWS_AGENTCORE_SUBNET_IDS - comma-separated subnet IDs
  - AWS_AGENTCORE_SG_IDS - comma-separated security group IDs

Note: Browser async e2e tests are omitted because Playwright's sync API (used
internally) conflicts with pytest-asyncio's event loop. The asyncio.to_thread()
delegation pattern is validated by the code interpreter async e2e tests instead.

Run:
    AWS_AGENTCORE_E2E=1 AWS_REGION=us-west-2 uv run pytest tests/test_browser_e2e.py -v
"""

import os
import random
import re
import string

import pytest

from llama_index.tools.aws_bedrock_agentcore.browser.base import (
    AgentCoreBrowserToolSpec,
)

E2E_ENABLED = os.environ.get("AWS_AGENTCORE_E2E")
SKIP_REASON = "Set AWS_AGENTCORE_E2E=1 and configure AWS credentials to run e2e tests"
ROLE_ARN = os.environ.get("AWS_AGENTCORE_ROLE_ARN")
SUBNET_IDS = os.environ.get("AWS_AGENTCORE_SUBNET_IDS")
SG_IDS = os.environ.get("AWS_AGENTCORE_SG_IDS")


@pytest.fixture(scope="module")
def tool_spec():
    """Create a shared Browser tool spec and clean up after all tests."""
    region = os.environ.get("AWS_REGION", "us-west-2")
    spec = AgentCoreBrowserToolSpec(region=region)
    yield spec
    for bc in spec._browser_clients.values():
        try:
            bc.stop()
        except Exception:
            pass
    spec._browser_clients.clear()


def _unique_name(prefix: str) -> str:
    """Generate a unique resource name with random suffix."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{suffix}"


def _extract_id(text: str) -> str:
    """Extract resource ID from response text like 'ID: some-id-123, Status: ...'."""
    match = re.search(r"ID:\s*([^,)]+)", text)
    return match.group(1).strip() if match else ""


@pytest.mark.skipif(not E2E_ENABLED, reason=SKIP_REASON)
class TestBrowserE2E:
    def test_navigate_and_current_webpage(self, tool_spec):
        nav_result = tool_spec.navigate_browser("https://example.com")
        assert "Navigated" in nav_result, (
            f"Expected navigation confirmation, got: {nav_result}"
        )

        info = tool_spec.current_webpage()
        assert "Example Domain" in info, (
            f"Expected 'Example Domain' in page info, got: {info}"
        )

    def test_extract_text(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        result = tool_spec.extract_text()
        assert "Example Domain" in result, (
            f"Expected 'Example Domain' in text, got: {result}"
        )

    def test_extract_hyperlinks(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        result = tool_spec.extract_hyperlinks()
        assert "iana.org" in result.lower(), (
            f"Expected iana.org link on example.com, got: {result}"
        )

    def test_get_elements(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        result = tool_spec.get_elements("h1")
        assert "Example Domain" in result, (
            f"Expected 'Example Domain' in h1 elements, got: {result}"
        )

    def test_click_element(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        result = tool_spec.click_element("a")
        assert "Error" not in result, f"Expected successful click, got: {result}"

    def test_navigate_back(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        tool_spec.navigate_browser("https://www.iana.org/domains/reserved")
        result = tool_spec.navigate_back()
        assert "Navigated back" in result or "example.com" in result.lower(), (
            f"Expected successful navigate back, got: {result}"
        )

    def test_generate_live_view_url(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        result = tool_spec.generate_live_view_url()
        assert "Error" not in result, f"Expected live view URL, got: {result}"
        assert "No browser session" not in result, (
            f"Expected active session for live view, got: {result}"
        )

    def test_list_browsers(self, tool_spec):
        result = tool_spec.list_browsers()
        assert "Error" not in result, f"Expected browser listing, got: {result}"
        assert "Found" in result or "No browsers" in result, (
            f"Unexpected listing result: {result}"
        )

    def test_take_and_release_control(self, tool_spec):
        tool_spec.navigate_browser("https://example.com")
        take_result = tool_spec.take_control()
        assert "manual control" in take_result.lower(), (
            f"Expected take control confirmation, got: {take_result}"
        )
        release_result = tool_spec.release_control()
        assert (
            "released" in release_result.lower()
            or "re-enabled" in release_result.lower()
        ), f"Expected release control confirmation, got: {release_result}"

    def test_create_get_delete_browser(self, tool_spec):
        if not ROLE_ARN:
            pytest.skip("AWS_AGENTCORE_ROLE_ARN not set")
        browser_id = None
        try:
            name = _unique_name("llama_e2e_browser")
            create_result = tool_spec.create_browser(
                name=name,
                execution_role_arn=ROLE_ARN,
                network_mode="PUBLIC",
                description="e2e test browser",
            )
            assert "Error" not in create_result, (
                f"Browser creation failed: {create_result}"
            )
            browser_id = _extract_id(create_result)
            assert browser_id, f"Could not extract browser ID from: {create_result}"

            get_result = tool_spec.get_browser(browser_id)
            assert "Error" not in get_result, f"Get browser failed: {get_result}"
            assert name in get_result, (
                f"Expected browser name in get result, got: {get_result}"
            )
        finally:
            if browser_id:
                delete_result = tool_spec.delete_browser(browser_id)
                assert "Error" not in delete_result, (
                    f"Delete browser failed: {delete_result}"
                )

    def test_create_browser_vpc(self, tool_spec):
        if not ROLE_ARN:
            pytest.skip("AWS_AGENTCORE_ROLE_ARN not set")
        if not SUBNET_IDS or not SG_IDS:
            pytest.skip("AWS_AGENTCORE_SUBNET_IDS and AWS_AGENTCORE_SG_IDS not set")
        subnet_ids = [s.strip() for s in SUBNET_IDS.split(",")]
        sg_ids = [s.strip() for s in SG_IDS.split(",")]
        browser_id = None
        try:
            create_result = tool_spec.create_browser(
                name=_unique_name("llama_e2e_browser_vpc"),
                execution_role_arn=ROLE_ARN,
                network_mode="VPC",
                subnet_ids=subnet_ids,
                security_group_ids=sg_ids,
            )
            assert "Error" not in create_result, (
                f"VPC browser creation failed: {create_result}"
            )
            browser_id = _extract_id(create_result)
            assert browser_id, f"Could not extract browser ID from: {create_result}"
        finally:
            if browser_id:
                tool_spec.delete_browser(browser_id)
