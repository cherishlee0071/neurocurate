"""Verified Phase 1.2 acquisition; NPY inspection never enables pickle."""
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import numpy as np
from .lowdisk import _download, sha256


def verify_artifact(artifact):
    path = Path(artifact["path"])
    if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise ValueError(f"size/SHA256 verification failed: {path.name}")


def acquire(artifact, work_dir, reserve_bytes):
    path = Path(artifact["path"])
    if not path.exists():
        if artifact["bytes"] + reserve_bytes > shutil.disk_usage(work_dir).free:
            raise ValueError("insufficient space for audited download and reserve")
        _download(artifact, work_dir)
    verify_artifact(artifact)  # Includes inputs from an earlier interrupted run.
    return dict(artifact, verified=True, actual_sha256=sha256(path))


def safe_member_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe ZIP member name")
    return path


def extract_member(archive, member, root, *, reserve_bytes=1024**3):
    """Bounded extraction to a new file. zipfile validates CRC at EOF.

Failed partials and preexisting files are retained; no overwrite or deletion.
"""
    relative = safe_member_name(member.filename)
    if member.is_dir() or member.flag_bits & 1 or (member.external_attr >> 16) & 0o170000 == 0o120000:
        raise ValueError("directory, encrypted member or symlink refused")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root.joinpath(*relative.parts)
    if root not in destination.resolve().parents:
        raise ValueError("member destination escapes controlled directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # Compare existing content with verified archive, without overwriting it.
        import hashlib
        digest = hashlib.sha256()
        with archive.open(member) as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
        if destination.stat().st_size != member.file_size or sha256(destination) != digest.hexdigest():
            raise ValueError("existing extracted file differs from verified archive")
        return destination
    if member.file_size + reserve_bytes > shutil.disk_usage(root).free:
        raise ValueError("insufficient space for member extraction and reserve")
    partial = destination.with_name(destination.name + ".partial")
    count = 0
    with archive.open(member) as source, partial.open("xb") as target:
        for chunk in iter(lambda: source.read(1024**2), b""):
            count += len(chunk)
            if count > member.file_size:
                raise ValueError("extraction exceeded declared member size")
            target.write(chunk)
    if count != member.file_size:
        raise ValueError("extracted size mismatch")
    if destination.exists():
        raise FileExistsError("extraction destination appeared")
    partial.rename(destination)
    return destination


def inspect_npy_header(path):
    """Read dtype and serialized array shape without deserializing objects."""
    with Path(path).open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError("unsupported NPY header version")
        header_bytes = source.tell()
    report = dict(path=str(path), npy_version=list(version), serialized_shape=list(shape),
                  serialized_dtype=str(dtype), fortran_order=fortran,
                  object_dtype=dtype.hasobject, header_bytes=header_bytes,
                  pickle_loaded=False)
    if dtype.hasobject:
        # np.load refuses at the header, before any pickle content is read.
        try:
            np.load(path, allow_pickle=False)
        except ValueError as error:
            report["allow_pickle_false_result"] = str(error)
        else:
            raise ValueError("object NPY unexpectedly loaded without pickle")
    else:
        report["allow_pickle_false_result"] = "numeric format compatible"
    return report
