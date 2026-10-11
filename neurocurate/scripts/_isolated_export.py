"""Unprivileged descriptor-only child. Install kernel confinement BEFORE pickle."""
import ctypes
import errno
import hashlib
import json
import os
import resource
import socket
import sys

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
# The 2022 pickle references legacy NumPy module names. Preload this trusted
# runtime code before confinement; filesystem opens remain forbidden afterward.
import numpy.core.multiarray
import numpy.core._multiarray_umath

assert len(sys.argv) in (8, 10)
fds = list(map(int, sys.argv[1:8]))
assert len(fds) == 7
inputs = [os.fdopen(fd, "rb") for fd in fds[:2]]
outputs = [os.fdopen(fd, "wb") for fd in fds[2:]]
uid = os.getuid()
assert uid != 0
assert len(os.listdir("/proc/self/task")) == 1, "single-threaded filter installation required"
open_fds = {int(x) for x in os.listdir("/proc/self/fd")}
# listdir temporarily opens one directory descriptor; it is closed on return.
for fd in open_fds - set(fds) - {0, 1, 2}:
    try:
        os.fstat(fd)
    except OSError:
        continue
    raise RuntimeError("unexpected inherited file descriptor")
lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x00050000 | errno.EPERM)  # default deny every syscall
allowed = ["read", "write", "readv", "writev", "close", "fstat", "lseek", "pread64", "pwrite64",
           "mmap", "mprotect", "munmap", "brk", "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
           "futex", "clock_gettime", "getpid", "gettid", "exit", "exit_group", "madvise",
           "sched_yield", "getrandom", "restart_syscall", "dup", "dup2", "dup3", "fcntl"]
for name in allowed:
    number = lib.seccomp_syscall_resolve_name(name.encode())
    assert number >= 0 and lib.seccomp_rule_add(ctx, 0x7fff0000, number, 0) == 0
assert lib.seccomp_load(ctx) == 0, "cannot install confinement; DO NOT LOAD"
probes = {}
for name, operation in [("filesystem_open", lambda: os.open("/etc/passwd", os.O_RDONLY)),
                        ("network_socket", lambda: socket.socket()), ("process_creation", os.fork)]:
    try:
        operation()
    except PermissionError:
        probes[name] = "denied"
    else:
        raise RuntimeError("confinement probe unexpectedly succeeded; DO NOT LOAD")
expected = ["08e52b18caf46b8504beb4edaadef025f114eaea25ab1f4eb8c5f9114c56ebff",
            "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c"]
if len(sys.argv) == 10:
    expected = sys.argv[8:10]
    assert all(len(value) == 64 and all(c in '0123456789abcdef' for c in value) for value in expected)
for handle, digest in zip(inputs, expected):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024**2), b""):
        hasher.update(chunk)
    assert hasher.hexdigest() == digest, "approved input identity mismatch"
    handle.seek(0)
eeg = np.load(inputs[0], allow_pickle=True).item()
metadata = np.load(inputs[1], allow_pickle=True).item()
assert type(eeg) is dict and set(eeg) == {"preprocessed_eeg_data", "ch_names", "times"}
array = eeg["preprocessed_eeg_data"]
assert type(array) is np.ndarray and array.shape == (200, 80, 17, 100)
assert array.dtype == np.dtype("float64") and np.isfinite(array).all()
times = np.asarray(eeg["times"])
channels = np.asarray(eeg["ch_names"], dtype=str)
assert times.shape == (100,) and times.dtype.kind == "f" and np.isfinite(times).all()
assert channels.shape == (17,) and len(set(channels)) == 17
assert type(metadata) is dict and all(type(k) is str for k in metadata)
clean_metadata = {}
for key, value in metadata.items():
    strings = np.asarray(value)
    assert strings.ndim == 1 and strings.dtype.kind in "US"
    clean_metadata[key] = strings.tolist()
for handle, values in zip(outputs[:3], [array, times, channels]):
    np.save(handle, values, allow_pickle=False)
    handle.flush()
outputs[3].write(json.dumps(clean_metadata, ensure_ascii=True).encode())
outputs[3].flush()
report = dict(uid=uid, root=False, single_threaded=True, network_access=False,
    confinement="kernel seccomp default-deny syscall allowlist, pre-opened descriptors only",
    filesystem_access="two readonly approved inputs; five designated writable outputs; stdio only",
    probes=probes, allowed_syscalls=allowed, address_space_limit_bytes=4*1024**3,
    cpu_limit_seconds=120, file_size_limit_bytes=512*1024**2, core_dumps=False,
    approved_sha256=expected, numeric_eeg_shape=list(array.shape), dtype=str(array.dtype),
    metadata_keys=sorted(clean_metadata), all_exports_pickle_free=True)
outputs[4].write(json.dumps(report, indent=2).encode())
for handle in outputs:
    handle.flush()
os._exit(0)
