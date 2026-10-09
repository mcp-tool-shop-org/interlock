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
        ("NVIDIA", PRESENT),
        ("HETERO:GPU.0,CPU", PRESENT),
        ("AUTO:GPU.0", PRESENT),
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


def _live(ov_id, full_name, device_type, architecture):
    return {
        "id": ov_id,
        "full_name": full_name,
        "device_type": device_type,
        "device_architecture": architecture,
    }


LIVE = [
    _live("NPU", "Intel(R) AI Boost", "INTEGRATED", "3720"),
    _live("CPU", "Intel(R) Core(TM) Ultra 9 285K", "INTEGRATED", "intel64"),
    _live("GPU.0", "Intel(R) Graphics (iGPU)", "INTEGRATED", "GPU: vendor=0x8086 arch=v12.70.4"),
    _live("GPU.1", "NVIDIA GeForce RTX 5090 (dGPU)", "DISCRETE", "GPU: vendor=0x10de arch=v12.0.0"),
]


def test_live_enumeration_picks_the_vendor_igpu_and_keeps_the_name():
    resolved = resolve_device("iGPU", LIVE)
    assert (resolved.kind, resolved.ov_id) == (IGPU, "GPU.0")
    assert resolved.full_name == "Intel(R) Graphics (iGPU)"
    by_id = resolve_device("GPU.0", LIVE)
    assert by_id.full_name == resolved.full_name


def test_a_renamed_igpu_still_resolves_from_vendor_and_type():
    drifted = _live("GPU.0", "Graphics", "integrated", "GPU: vendor=0x8086 arch=v12.70.4")
    resolved = resolve_device("iGPU", [drifted])
    assert resolved.kind == IGPU
    assert resolved.full_name == "Graphics"


def test_an_intel_name_is_not_enough_when_type_or_vendor_disagree():
    discrete = _live("GPU.0", "Intel(R) Graphics (iGPU)", "DISCRETE", "GPU: vendor=0x8086 arch=v12")
    other_vendor = _live("GPU.0", "Intel(R) Graphics (iGPU)", "INTEGRATED", "GPU: vendor=0x10de arch=v12")
    no_vendor = _live("GPU.0", "Intel(R) Graphics (iGPU)", "INTEGRATED", "3720")
    for entry in (discrete, other_vendor, no_vendor):
        with pytest.raises(DeviceRefused, match="not an Intel iGPU"):
            resolve_device("GPU.0", [entry])
        with pytest.raises(DeviceRefused, match="no Intel iGPU"):
            resolve_device("iGPU", [entry, NPU_DEV])


def test_a_null_property_is_treated_as_absent():
    entry = {"id": "GPU.0", "full_name": "Intel(R) Graphics", "device_type": None}
    resolved = resolve_device("GPU.0", [entry])
    assert resolved.kind == IGPU


def test_a_partial_property_set_is_refused():
    partial = {"id": "GPU.0", "full_name": "Intel(R) Graphics (iGPU)", "device_type": "INTEGRATED"}
    with pytest.raises(DeviceRefused, match="needs both"):
        resolve_device("iGPU", [partial])
    blank = {"id": "GPU.0", "full_name": "Intel(R) Graphics (iGPU)", "DEVICE_ARCHITECTURE": ""}
    with pytest.raises(DeviceRefused, match="must be strings"):
        resolve_device("GPU.0", [blank])


def test_gpu_1_stays_refused_when_its_vendor_says_intel():
    spoofed = _live("GPU.1", "Intel(R) Graphics", "INTEGRATED", "GPU: vendor=0x8086 arch=v12")
    with pytest.raises(DeviceRefused, match="GPU.1 is refused"):
        resolve_device("GPU.1", [spoofed])
    with pytest.raises(DeviceRefused, match="no Intel iGPU"):
        resolve_device("iGPU", [spoofed])


def test_a_nvidia_gpu_0_is_not_chosen_as_the_igpu():
    with pytest.raises(DeviceRefused, match="no Intel iGPU"):
        resolve_device("iGPU", [NVIDIA_0, NPU_DEV])


def test_two_vendor_igpus_are_not_guessed():
    second = _live("GPU.2", "Graphics", "INTEGRATED", "GPU: vendor=0x8086 arch=v12")
    first = LIVE[2]
    with pytest.raises(DeviceRefused, match="more than one"):
        resolve_device("iGPU", [first, second])
    named = resolve_device("GPU.2", [first, second])
    assert named.ov_id == "GPU.2"
