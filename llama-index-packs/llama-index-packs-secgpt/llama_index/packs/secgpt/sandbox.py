"""
Each spoke runs in an isolated process. We leverage the seccomp and setrlimit system utilities to restrict access to system calls and set limits on the resources a process can consume. To implement them, we define several helper functions here, which can be configured to meet specific security or system requirements for different use scenarios or apps.
"""

import resource
import tldextract
import platform

# Set timeout for spoke execution
TIMEOUT = 180

# Set the memory, cpu and write limits
# These are app-specific and can be be adjusted as needed
MEMORY_LIMIT = resource.getrlimit(resource.RLIMIT_AS)[1]  # 10240 * 1024 * 1024
CPU_TIME_LIMIT = resource.getrlimit(resource.RLIMIT_CPU)[1]  # 2 * 60
WRITE_LIMIT = resource.getrlimit(resource.RLIMIT_FSIZE)[1]  # 10240 * 1024 * 1024

# Set the allowed root domains
# This is a list of root domains (eTLD+1) that the app is allowed to access
allowed_domains = ["localhost"]


def get_root_domain(url):
    """
    Extract the root domain from a given URL.

    Args:
        url (str): The URL to extract the root domain from.

    Returns:
        str: The root domain of the URL.

    """
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}"


def is_request_allowed(url):
    """
    Check if a request to a given URL is allowed based on the root domain.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the request is allowed, False otherwise.

    """
    root_domain = get_root_domain(url)
    return root_domain in allowed_domains


def set_mem_limit():
    """
    Set the CPU time, maximum virtual memory, and write limits for the process.
    """
    # virtual memory
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    # cpu time
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_TIME_LIMIT, CPU_TIME_LIMIT))
    # write limit i.e. don't allow an infinite stream to stdout/stderr
    resource.setrlimit(resource.RLIMIT_FSIZE, (WRITE_LIMIT, WRITE_LIMIT))


# seccomp only works for Linux
if platform.system() == "Linux":
    import pyseccomp as seccomp

    def drop_perms():
        """
        Set restrictions on system calls using seccomp for Linux.
        The restrictions can be adjusted as needed based on the app's specifications.
        """
        # Create a SyscallFilter instance with ALLOW as the default action
        filter = seccomp.SyscallFilter(seccomp.ALLOW)

        # load the filter in the kernel
        filter.load()

else:

    def drop_perms():
        """
        Define a placeholder function for non-Linux platforms to restrict system calls.
        """
