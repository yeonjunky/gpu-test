"""Resource limits applied inside the untrusted-code subprocess (exec_worker.py).

Belt-and-suspenders alongside the parent process's subprocess.run(timeout=...):
RLIMIT_CPU bounds runaway CPU-bound loops even if wall-clock timeout handling
in the parent is delayed for any reason; RLIMIT_AS bounds memory blowups.
"""
import resource

DEFAULT_MEMORY_BYTES = 1 * 1024 ** 3  # 1 GB
DEFAULT_CPU_SECONDS = 8


def apply_limits(memory_bytes: int = DEFAULT_MEMORY_BYTES, cpu_seconds: int = DEFAULT_CPU_SECONDS) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # cap number of subprocesses/threads the generated code could spawn
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (ValueError, resource.error):
        pass  # not settable in some sandboxed/container environments
