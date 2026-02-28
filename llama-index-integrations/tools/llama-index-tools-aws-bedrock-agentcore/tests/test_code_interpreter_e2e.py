"""
End-to-end integration tests for AgentCoreCodeInterpreterToolSpec.

These tests hit real AgentCore CodeInterpreter services and require:
  - Valid AWS credentials with AgentCore access
  - AWS_AGENTCORE_E2E=1 environment variable

Lifecycle tests (create/get/delete) additionally require:
  - AWS_AGENTCORE_ROLE_ARN - IAM role ARN for interpreter operations

VPC lifecycle tests additionally require:
  - AWS_AGENTCORE_SUBNET_IDS - comma-separated subnet IDs
  - AWS_AGENTCORE_SG_IDS - comma-separated security group IDs

Run:
    AWS_AGENTCORE_E2E=1 AWS_REGION=us-west-2 uv run pytest tests/test_code_interpreter_e2e.py -v
"""

import asyncio
import os
import random
import re
import string

import pytest

from llama_index.tools.aws_bedrock_agentcore.code_interpreter.base import (
    AgentCoreCodeInterpreterToolSpec,
)

E2E_ENABLED = os.environ.get("AWS_AGENTCORE_E2E")
SKIP_REASON = "Set AWS_AGENTCORE_E2E=1 and configure AWS credentials to run e2e tests"
ROLE_ARN = os.environ.get("AWS_AGENTCORE_ROLE_ARN")
SUBNET_IDS = os.environ.get("AWS_AGENTCORE_SUBNET_IDS")
SG_IDS = os.environ.get("AWS_AGENTCORE_SG_IDS")


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def tool_spec():
    """Create a shared CodeInterpreter tool spec and clean up after all tests."""
    region = os.environ.get("AWS_REGION", "us-west-2")
    spec = AgentCoreCodeInterpreterToolSpec(region=region)
    yield spec
    for ci in spec._code_interpreters.values():
        try:
            ci.stop()
        except Exception:
            pass
    spec._code_interpreters.clear()


@pytest.fixture(scope="module")
def async_tool_spec():
    """Create a shared CodeInterpreter tool spec for async tests."""
    region = os.environ.get("AWS_REGION", "us-west-2")
    spec = AgentCoreCodeInterpreterToolSpec(region=region)
    yield spec
    for ci in spec._code_interpreters.values():
        try:
            ci.stop()
        except Exception:
            pass
    spec._code_interpreters.clear()


def _unique_name(prefix: str) -> str:
    """Generate a unique resource name with random suffix."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{suffix}"


def _extract_id(text: str) -> str:
    """Extract resource ID from response text like 'ID: some-id-123, Status: ...'."""
    match = re.search(r"ID:\s*([^,)]+)", text)
    return match.group(1).strip() if match else ""


@pytest.mark.skipif(not E2E_ENABLED, reason=SKIP_REASON)
class TestCodeInterpreterE2E:
    def test_execute_code(self, tool_spec):
        result = tool_spec.execute_code("print(1+1)")
        assert "2" in result, f"Expected '2' in output, got: {result}"

    def test_execute_command(self, tool_spec):
        result = tool_spec.execute_command("echo hello")
        assert "hello" in result, f"Expected 'hello' in output, got: {result}"

    def test_write_and_read_files(self, tool_spec):
        content = "e2e test content"
        tool_spec.write_files([{"path": "e2e_test.txt", "text": content}])
        result = tool_spec.read_files(["e2e_test.txt"])
        assert content in result, (
            f"Expected written content in read result, got: {result}"
        )

    def test_list_files(self, tool_spec):
        tool_spec.write_files([{"path": "e2e_list_test.txt", "text": "list test"}])
        result = tool_spec.list_files()
        assert "Error" not in result, f"list_files returned an error: {result}"

    def test_delete_files(self, tool_spec):
        tool_spec.write_files([{"path": "e2e_delete_me.txt", "text": "delete me"}])
        tool_spec.delete_files(["e2e_delete_me.txt"])
        result = tool_spec.execute_command("ls e2e_delete_me.txt 2>&1")
        assert "No such file" in result or "e2e_delete_me.txt" not in result, (
            f"Expected file to be deleted, got: {result}"
        )

    def test_upload_and_download_file(self, tool_spec):
        content = "uploaded via upload_file"
        upload_result = tool_spec.upload_file(
            path="e2e_upload.txt", content=content, description="e2e test file"
        )
        assert "Uploaded" in upload_result, (
            f"Expected upload confirmation, got: {upload_result}"
        )

        download_result = tool_spec.download_file(path="e2e_upload.txt")
        assert content in download_result, (
            f"Expected uploaded content in download, got: {download_result}"
        )

    def test_upload_files_batch(self, tool_spec):
        files = [
            {"path": "e2e_batch_1.txt", "content": "batch file one"},
            {"path": "e2e_batch_2.txt", "content": "batch file two"},
        ]
        upload_result = tool_spec.upload_files(files)
        assert "2" in upload_result, (
            f"Expected '2' files uploaded, got: {upload_result}"
        )

        dl1 = tool_spec.download_file(path="e2e_batch_1.txt")
        dl2 = tool_spec.download_file(path="e2e_batch_2.txt")
        assert "batch file one" in dl1, f"Expected batch file 1 content, got: {dl1}"
        assert "batch file two" in dl2, f"Expected batch file 2 content, got: {dl2}"

    def test_download_files_batch(self, tool_spec):
        tool_spec.upload_file(path="e2e_dl_batch_1.txt", content="dl batch one")
        tool_spec.upload_file(path="e2e_dl_batch_2.txt", content="dl batch two")
        result = tool_spec.download_files(["e2e_dl_batch_1.txt", "e2e_dl_batch_2.txt"])
        assert "dl batch one" in result, (
            f"Expected file 1 content in batch download, got: {result}"
        )
        assert "dl batch two" in result, (
            f"Expected file 2 content in batch download, got: {result}"
        )

    def test_install_packages(self, tool_spec):
        install_result = tool_spec.install_packages(["requests"])
        assert "Error" not in install_result, (
            f"Package install failed: {install_result}"
        )

        result = tool_spec.execute_code("import requests; print(requests.__version__)")
        assert "Error" not in result, f"Expected version string, got: {result}"
        assert result.strip(), f"Expected non-empty version output, got: {result}"

    def test_clear_context(self, tool_spec):
        tool_spec.execute_code("x = 42")
        clear_result = tool_spec.clear_context()
        assert "cleared" in clear_result.lower(), (
            f"Expected clear confirmation, got: {clear_result}"
        )

        result = tool_spec.execute_code("print(x)")
        assert "NameError" in result, (
            f"Expected NameError after clearing context, got: {result}"
        )

    def test_start_get_stop_task(self, tool_spec):
        start_result = tool_spec.start_command("sleep 30 && echo done")
        assert "Error" not in start_result, (
            f"Expected task start result, got: {start_result}"
        )
        task_id = re.search(r"[a-f0-9-]{8,}", start_result)
        if task_id:
            get_result = tool_spec.get_task(task_id.group(0))
            assert "Error" not in get_result, f"Expected task info, got: {get_result}"
            stop_result = tool_spec.stop_task(task_id.group(0))
            assert "Error" not in stop_result, (
                f"Expected task stop result, got: {stop_result}"
            )

    def test_list_code_interpreters(self, tool_spec):
        result = tool_spec.list_code_interpreters()
        assert "Error" not in result, f"Expected interpreter listing, got: {result}"
        assert "Found" in result or "No code interpreters" in result, (
            f"Unexpected listing result: {result}"
        )

    def test_multi_thread(self, tool_spec):
        tool_spec.execute_code("thread_var = 'alpha'", thread_id="thread_a")
        tool_spec.execute_code("thread_var = 'beta'", thread_id="thread_b")

        result_a = tool_spec.execute_code("print(thread_var)", thread_id="thread_a")
        result_b = tool_spec.execute_code("print(thread_var)", thread_id="thread_b")

        assert "alpha" in result_a, f"Expected 'alpha' on thread_a, got: {result_a}"
        assert "beta" in result_b, f"Expected 'beta' on thread_b, got: {result_b}"

    def test_create_get_delete_code_interpreter(self, tool_spec):
        if not ROLE_ARN:
            pytest.skip("AWS_AGENTCORE_ROLE_ARN not set")
        interpreter_id = None
        try:
            name = _unique_name("llama_e2e_ci")
            create_result = tool_spec.create_code_interpreter(
                name=name,
                execution_role_arn=ROLE_ARN,
                network_mode="PUBLIC",
                description="e2e test interpreter",
            )
            assert "Error" not in create_result, (
                f"Interpreter creation failed: {create_result}"
            )
            interpreter_id = _extract_id(create_result)
            assert interpreter_id, (
                f"Could not extract interpreter ID from: {create_result}"
            )

            get_result = tool_spec.get_code_interpreter(interpreter_id)
            assert "Error" not in get_result, f"Get interpreter failed: {get_result}"
            assert name in get_result, (
                f"Expected interpreter name in get result, got: {get_result}"
            )
        finally:
            if interpreter_id:
                delete_result = tool_spec.delete_code_interpreter(interpreter_id)
                assert "Error" not in delete_result, (
                    f"Delete interpreter failed: {delete_result}"
                )

    def test_create_code_interpreter_vpc(self, tool_spec):
        if not ROLE_ARN:
            pytest.skip("AWS_AGENTCORE_ROLE_ARN not set")
        if not SUBNET_IDS or not SG_IDS:
            pytest.skip("AWS_AGENTCORE_SUBNET_IDS and AWS_AGENTCORE_SG_IDS not set")
        subnet_ids = [s.strip() for s in SUBNET_IDS.split(",")]
        sg_ids = [s.strip() for s in SG_IDS.split(",")]
        interpreter_id = None
        try:
            create_result = tool_spec.create_code_interpreter(
                name=_unique_name("llama_e2e_ci_vpc"),
                execution_role_arn=ROLE_ARN,
                network_mode="VPC",
                subnet_ids=subnet_ids,
                security_group_ids=sg_ids,
            )
            assert "Error" not in create_result, (
                f"VPC interpreter creation failed: {create_result}"
            )
            interpreter_id = _extract_id(create_result)
            assert interpreter_id, (
                f"Could not extract interpreter ID from: {create_result}"
            )
        finally:
            if interpreter_id:
                tool_spec.delete_code_interpreter(interpreter_id)


@pytest.mark.skipif(not E2E_ENABLED, reason=SKIP_REASON)
class TestCodeInterpreterAsyncE2E:
    @pytest.mark.asyncio
    async def test_async_execute_code(self, async_tool_spec):
        result = await async_tool_spec.aexecute_code("print(1+1)")
        assert "2" in result, f"Expected '2' in async output, got: {result}"

    @pytest.mark.asyncio
    async def test_async_list_code_interpreters(self, async_tool_spec):
        result = await async_tool_spec.alist_code_interpreters()
        assert "Error" not in result, (
            f"Expected async interpreter listing, got: {result}"
        )
        assert "Found" in result or "No code interpreters" in result, (
            f"Unexpected async listing result: {result}"
        )

    @pytest.mark.asyncio
    async def test_async_execute_command(self, async_tool_spec):
        result = await async_tool_spec.aexecute_command("echo hello")
        assert "hello" in result, f"Expected 'hello' in async output, got: {result}"

    @pytest.mark.asyncio
    async def test_async_write_and_read_files(self, async_tool_spec):
        content = "async e2e test content"
        await async_tool_spec.awrite_files(
            [{"path": "async_e2e_test.txt", "text": content}]
        )
        result = await async_tool_spec.aread_files(["async_e2e_test.txt"])
        assert content in result, (
            f"Expected written content in async read result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_async_list_files(self, async_tool_spec):
        await async_tool_spec.awrite_files(
            [{"path": "async_e2e_list_test.txt", "text": "async list test"}]
        )
        result = await async_tool_spec.alist_files()
        assert "Error" not in result, f"async list_files returned an error: {result}"

    @pytest.mark.asyncio
    async def test_async_delete_files(self, async_tool_spec):
        await async_tool_spec.awrite_files(
            [{"path": "async_e2e_delete_me.txt", "text": "delete me"}]
        )
        await async_tool_spec.adelete_files(["async_e2e_delete_me.txt"])
        result = await async_tool_spec.aexecute_command(
            "ls async_e2e_delete_me.txt 2>&1"
        )
        assert "No such file" in result or "async_e2e_delete_me.txt" not in result, (
            f"Expected file to be deleted, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_async_start_get_stop_task(self, async_tool_spec):
        start_result = await async_tool_spec.astart_command("sleep 30 && echo done")
        assert "Error" not in start_result, (
            f"Expected async task start result, got: {start_result}"
        )
        task_id = re.search(r"[a-f0-9-]{8,}", start_result)
        if task_id:
            get_result = await async_tool_spec.aget_task(task_id.group(0))
            assert "Error" not in get_result, (
                f"Expected async task info, got: {get_result}"
            )
            stop_result = await async_tool_spec.astop_task(task_id.group(0))
            assert "Error" not in stop_result, (
                f"Expected async task stop result, got: {stop_result}"
            )

    @pytest.mark.asyncio
    async def test_async_upload_and_download_file(self, async_tool_spec):
        content = "async uploaded via aupload_file"
        upload_result = await async_tool_spec.aupload_file(
            path="async_e2e_upload.txt", content=content, description="async e2e file"
        )
        assert "Uploaded" in upload_result, (
            f"Expected async upload confirmation, got: {upload_result}"
        )

        download_result = await async_tool_spec.adownload_file(
            path="async_e2e_upload.txt"
        )
        assert content in download_result, (
            f"Expected uploaded content in async download, got: {download_result}"
        )

    @pytest.mark.asyncio
    async def test_async_upload_files_batch(self, async_tool_spec):
        files = [
            {"path": "async_e2e_batch_1.txt", "content": "async batch file one"},
            {"path": "async_e2e_batch_2.txt", "content": "async batch file two"},
        ]
        upload_result = await async_tool_spec.aupload_files(files)
        assert "2" in upload_result, (
            f"Expected '2' files uploaded async, got: {upload_result}"
        )

        dl1 = await async_tool_spec.adownload_file(path="async_e2e_batch_1.txt")
        dl2 = await async_tool_spec.adownload_file(path="async_e2e_batch_2.txt")
        assert "async batch file one" in dl1, (
            f"Expected async batch file 1 content, got: {dl1}"
        )
        assert "async batch file two" in dl2, (
            f"Expected async batch file 2 content, got: {dl2}"
        )

    @pytest.mark.asyncio
    async def test_async_download_files_batch(self, async_tool_spec):
        await async_tool_spec.aupload_file(
            path="async_e2e_dl_batch_1.txt", content="async dl batch one"
        )
        await async_tool_spec.aupload_file(
            path="async_e2e_dl_batch_2.txt", content="async dl batch two"
        )
        result = await async_tool_spec.adownload_files(
            ["async_e2e_dl_batch_1.txt", "async_e2e_dl_batch_2.txt"]
        )
        assert "async dl batch one" in result, (
            f"Expected file 1 content in async batch download, got: {result}"
        )
        assert "async dl batch two" in result, (
            f"Expected file 2 content in async batch download, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_async_install_packages(self, async_tool_spec):
        install_result = await async_tool_spec.ainstall_packages(["requests"])
        assert "Error" not in install_result, (
            f"Async package install failed: {install_result}"
        )

        result = await async_tool_spec.aexecute_code(
            "import requests; print(requests.__version__)"
        )
        assert "Error" not in result, (
            f"Expected version string in async result, got: {result}"
        )
        assert result.strip(), (
            f"Expected non-empty version output in async result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_async_clear_context(self, async_tool_spec):
        await async_tool_spec.aexecute_code("x = 42")
        clear_result = await async_tool_spec.aclear_context()
        assert "cleared" in clear_result.lower(), (
            f"Expected async clear confirmation, got: {clear_result}"
        )

        result = await async_tool_spec.aexecute_code("print(x)")
        assert "NameError" in result, (
            f"Expected NameError after async clearing context, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_async_lifecycle(self, async_tool_spec):
        if not ROLE_ARN:
            pytest.skip("AWS_AGENTCORE_ROLE_ARN not set")
        interpreter_id = None
        try:
            create_result = await async_tool_spec.acreate_code_interpreter(
                name=_unique_name("llama_e2e_ci_async"),
                execution_role_arn=ROLE_ARN,
                network_mode="PUBLIC",
                description="async e2e test interpreter",
            )
            assert "Error" not in create_result, (
                f"Async interpreter creation failed: {create_result}"
            )
            interpreter_id = _extract_id(create_result)
            assert interpreter_id, (
                f"Could not extract interpreter ID from: {create_result}"
            )

            get_result = await async_tool_spec.aget_code_interpreter(interpreter_id)
            assert "Error" not in get_result, (
                f"Async get interpreter failed: {get_result}"
            )

            delete_result = await async_tool_spec.adelete_code_interpreter(
                interpreter_id
            )
            assert "Error" not in delete_result, (
                f"Async delete interpreter failed: {delete_result}"
            )
            interpreter_id = None
        finally:
            if interpreter_id:
                await async_tool_spec.adelete_code_interpreter(interpreter_id)
