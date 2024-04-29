import sys
import resource
import tldextract
import platform

# Set timeout for spoke execution
TIMEOUT = 180

# Set the memory, cpu and write limits
# These are app-specific and can be be adjusted as needed
MEMORY_LIMIT = resource.getrlimit(resource.RLIMIT_AS)[1] #10240 * 1024 * 1024  
CPU_TIME_LIMIT = resource.getrlimit(resource.RLIMIT_CPU)[1] #2 * 60  
WRITE_LIMIT = resource.getrlimit(resource.RLIMIT_FSIZE)[1] #10240 * 1024 * 1024  

# Set the allowed root domains
# This is a list of root domains (eTLD+1) that the app is allowed to access 
allowed_domains = [
    "localhost"
]

def get_root_domain(url):
    extracted = tldextract.extract(url)
    root_domain = "{}.{}".format(extracted.domain, extracted.suffix)
    return root_domain

def is_request_allowed(url):
    root_domain = get_root_domain(url)
    return root_domain in allowed_domains


# Set the CPU time, maximum virtual memory and write limits
def set_mem_limit():
    # virtual memory
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    # cpu time
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_TIME_LIMIT, CPU_TIME_LIMIT))
    # write limit i.e. don't allow an infinite stream to stdout/stderr
    resource.setrlimit(resource.RLIMIT_FSIZE, (WRITE_LIMIT, WRITE_LIMIT))

# seccomp only works for Linux
if platform.system() == "Linux":
    import pyseccomp as seccomp
    # Set restrictions on system calls
    # The restrictions can be adjusted as needed based on the app's specifications
    def drop_perms():
        # Create a SyscallFilter instance with ALLOW as the default action
        filter = seccomp.SyscallFilter(seccomp.ALLOW)

        # load the filter in the kernel
        filter.load()

else:
    # Can define methods to restrict system calls for other platforms
    def drop_perms():
        pass
