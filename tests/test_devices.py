import pytest

from switchyard.devices import CPU, IGPU, NPU, DeviceRefused, resolve_device

INTEL_GPU = {"id": "GPU.0", "full_name": "Intel(R) Graphics (iGPU)"}
NVIDIA_0 = {"id": "GPU.0", "full_name": "NVIDIA GeForce RTX 5090"}
NVIDIA_1 = {"id": "GPU.1", "full_name": "NVIDIA GeForce RTX 5090"}
SPOOFED_1 = {"id": "GPU.1", "full_name": "Intel(R) Graphics"}
NPU_DEV = {"id": "NPU", "full_name": "Intel(R) AI Boost"}
CPU_DEV = {"id": "CPU", "full_name": "Intel(R) Core(TM) Ultra 9 285K"}
PRESENT = [NPU_DEV, CPU_DEV, INTEL_GPU, NVIDIA_1]


def test_named_devices_resolve():
    npu = resolve_device("NPU", PRESENT)
    cpu = resolve_device("CPU", PRESENT)
    igpu = resolve_device("iGPU", PRESENT)
    assert (npu.kind, npu.ov_id) == (NPU, "NPU")
    assert (cpu.kind, cpu.ov_id) == (CPU, "CPU")
    assert (igpu.kind, igpu.ov_id, igpu.full_name) == (IGPU, "GPU.0", INTEL_GPU["full_name"])


def test_gpu_0_intel_is_the_igpu():
    resolved = resolve_device("GPU.0", [INTEL_GPU])
    assert resolved.kind == IGPU
    assert resolved.ov_id == "GPU.0"


@pytest.mark.parametrize(
    "requested, available",
    [
        ("GPU.1", [SPOOFED_1]),
        ("gpu.1", [NVIDIA_1]),
        ("GPU.1", PRESENT),
        ("GPU.0", [NVIDIA_0]),
        ("GPU.0", [NVIDIA_0, NVIDIA_1]),
        ("AUTO", PRESENT),
        ("auto", PRESENT),
        ("HETERO", PRESENT),
        ("MULTI:GPU.0,CPU", PRESENT),
        ("GPU", PRESENT),
        ("gpu", [INTEL_GPU]),
        ("5090", PRESENT),
        ("CUDA", PRESENT),
    ],
)
def test_resolver_refuses(requested, available):
    with pytest.raises(DeviceRefused):
        resolve_device(requested, available)


def test_missing_device_does_not_fall_back():
    with pytest.raises(DeviceRefused, match="not in the device list"):
        resolve_device("NPU", [CPU_DEV, INTEL_GPU])


def test_two_intel_gpus_are_not_guessed():
    second = {"id": "GPU.2", "full_name": "Intel(R) Arc Graphics"}
    with pytest.raises(DeviceRefused, match="more than one"):
        resolve_device("iGPU", [INTEL_GPU, second])
    named = resolve_device("GPU.2", [INTEL_GPU, second])
    assert named.ov_id == "GPU.2"


def test_malformed_entries_and_unknown_names_are_refused():
    with pytest.raises(DeviceRefused):
        resolve_device("", PRESENT)
    with pytest.raises(DeviceRefused, match="mapping"):
        resolve_device("NPU", ["NPU"])
    with pytest.raises(DeviceRefused, match="duplicate"):
        resolve_device("NPU", [NPU_DEV, {"id": "npu", "full_name": "again"}])
    with pytest.raises(DeviceRefused, match="unknown device"):
        resolve_device("TPU", PRESENT)
    with pytest.raises(DeviceRefused, match="not in the device list"):
        resolve_device("GPU.3", PRESENT)


def test_gpu_1_is_never_the_igpu_even_if_the_name_says_intel():
    with pytest.raises(DeviceRefused, match="no Intel iGPU"):
        resolve_device("iGPU", [SPOOFED_1, NPU_DEV])
