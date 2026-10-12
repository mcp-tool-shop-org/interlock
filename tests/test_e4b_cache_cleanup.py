import os
import stat

from switchyard.harness import e4b


def test_a_read_only_compiled_blob_is_removed(tmp_path):
    # Live 2026-10-12: OpenVINO's NPU blob is read-only and rmtree raised WinError 5 in the block's finally.
    cache = tmp_path / "compile-cache" / "run" / "npu-alone"
    cache.mkdir(parents=True)
    blob = cache / "1.blob"
    blob.write_bytes(b"x")
    os.chmod(blob, stat.S_IREAD)
    assert e4b._delete_caches(tmp_path, ["compile-cache/run/npu-alone"]) == []
    assert not cache.exists()


def test_a_cache_that_will_not_go_is_reported_not_raised(tmp_path, monkeypatch):
    cache = tmp_path / "compile-cache" / "run" / "npu-alone"
    cache.mkdir(parents=True)

    def refuse(path, onexc=None):
        raise PermissionError(5, "Access is denied", str(path))

    monkeypatch.setattr(e4b.shutil, "rmtree", refuse)
    failed = e4b._delete_caches(tmp_path, ["compile-cache/run/npu-alone", "compile-cache/run/missing"])
    assert len(failed) == 1 and failed[0].startswith("compile-cache/run/npu-alone: ")
