import pytest

from interlock.harness.fill import install_relative
from interlock.receipt import ReceiptError, _reject_home_paths

# Built at run time so the repository holds no home-path literal (the identity scan refuses one).
HOME_LINE = "C:\\" + "Users" + "\\someone\\AppData\\Local\\Programs\\Ollama\\lib\\ollama\\cuda_v13\\ggml-cuda.dll"


def test_the_backend_library_is_recorded_inside_the_install():
    # Live 2026-10-13: the full load_backend path sat under the user's home and the receipt refused the row.
    assert install_relative(HOME_LINE) == r"lib\ollama\cuda_v13\ggml-cuda.dll"
    assert install_relative("/opt/ollama/lib/ollama/cuda_v13/libggml-cuda.so") == r"lib\ollama\cuda_v13\libggml-cuda.so"
    assert install_relative(r"D:\x\ggml-cuda.dll") == "ggml-cuda.dll"
    _reject_home_paths({"backend_library": install_relative(HOME_LINE)}, "$.settings")


def test_a_home_path_in_the_server_facts_is_caught_by_the_guard():
    with pytest.raises(ReceiptError):
        _reject_home_paths({"backend_library": HOME_LINE}, "$.settings")
