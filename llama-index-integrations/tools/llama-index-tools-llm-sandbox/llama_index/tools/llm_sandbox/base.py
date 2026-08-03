"""LLM Sandbox tool spec."""

from typing import Any, Dict, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

# Dropping every capability breaks execution on its own: the session copies the
# source file into the container, and without DAC_OVERRIDE it cannot be read.
# Everything else stays dropped -- CapEff is 0000000000000002 inside.
#
# read_only is deliberately absent: Docker rejects the code copy against a
# read-only rootfs, with or without a tmpfs on the workdir.
DEFAULT_RUNTIME_CONFIGS: Dict[str, Any] = {
    "network_mode": "none",
    "mem_limit": "512m",
    "pids_limit": 128,
    "cap_drop": ["ALL"],
    "cap_add": ["DAC_OVERRIDE"],
    "security_opt": ["no-new-privileges:true"],
}


class LLMSandboxToolSpec(BaseToolSpec):
    """
    LLM Sandbox tool spec.

    Runs agent-authored code inside an isolated container rather than on the
    host, using `llm-sandbox <https://github.com/vndee/llm-sandbox>`_. Supports
    Docker, Podman and Kubernetes backends and seven languages.

    By default the container has no network access, capped memory and pids, all
    Linux capabilities dropped except DAC_OVERRIDE, and no-new-privileges set.
    Pass ``runtime_configs`` to change that.

    Note that this is container isolation, not VM isolation: it inherits the
    threat model of the backend in use and makes no kernel-level guarantee. For
    deliberately adversarial code, pair it with a hardened runtime such as
    gVisor or Kata Containers.
    """

    spec_functions = ["execute_code"]

    def __init__(
        self,
        lang: str = "python",
        backend: str = "docker",
        image: Optional[str] = None,
        timeout: float = 30.0,
        runtime_configs: Optional[Dict[str, Any]] = None,
        keep_template: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the tool spec.

        Args:
            lang: Language to execute. One of python, javascript, java, cpp, go,
                r, ruby.
            backend: Container backend: docker, podman, kubernetes or micromamba.
            image: Optional custom image. The tool deliberately exposes no
                package-installation parameter: the default network isolation
                would block it, and letting a model choose arbitrary PyPI
                packages runs setup.py code at install time. Pre-bake
                dependencies into an image instead.
            timeout: Seconds before an execution is aborted.
            runtime_configs: Container settings passed to the backend. Defaults
                to ``DEFAULT_RUNTIME_CONFIGS``; pass a dict to replace it.
            keep_template: Keep the base image after the session closes. Leaving
                this False re-pulls the image on every call.
            verbose: Emit session logs.

        """
        self.lang = lang
        self.backend = backend
        self.image = image
        self.timeout = timeout
        self.runtime_configs = DEFAULT_RUNTIME_CONFIGS if runtime_configs is None else runtime_configs
        self.keep_template = keep_template
        self.verbose = verbose

    def _session_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "lang": self.lang,
            "backend": self.backend,
            "verbose": self.verbose,
            "keep_template": self.keep_template,
            "runtime_configs": self.runtime_configs,
        }
        if self.image is not None:
            kwargs["image"] = self.image
        return kwargs

    def execute_code(self, code: str) -> str:
        """
        Execute code in an isolated container and return what it printed.

        Use this for calculations, data manipulation, and anything easier to
        compute than to reason about. Always print the result: only stdout is
        returned.

        Args:
            code: Source to execute, complete and self-contained.

        Returns:
            stdout on success. On a non-zero exit, the exit code followed by
            stderr, or stdout when stderr is empty.

        """
        from llm_sandbox import SandboxSession

        with SandboxSession(**self._session_kwargs()) as session:
            result = session.run(code, timeout=self.timeout)

        if result.exit_code != 0:
            return f"exit {result.exit_code}\n{result.stderr or result.stdout}".strip()
        return result.stdout.strip() or "(no output)"
