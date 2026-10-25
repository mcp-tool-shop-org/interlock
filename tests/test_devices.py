import pytest

from interlock.devices import CPU, IGPU, NPU, DeviceRefused, resolve_device

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


def test_remote_colab_cards_and_identity_validation():
    from interlock.devices import (
        COLAB_CARDS,
        REMOTE_DEVICES,
        REMOTE_PROVIDERS,
        RemoteDeviceIdentity,
        parse_remote_device,
    )

    assert REMOTE_PROVIDERS == ("colab",)
    assert COLAB_CARDS == ("A100-40", "A100-80", "L4", "T4", "G4")
    assert REMOTE_DEVICES == (
        "colab:A100-40",
        "colab:A100-80",
        "colab:L4",
        "colab:T4",
        "colab:G4",
    )
    for card in COLAB_CARDS:
        ident = RemoteDeviceIdentity(
            provider="colab",
            card=card,
            vram_mib=40960,
            driver_version="550.54.15",
            cuda_version="12.4",
            pool="us-central1-a",
        )
        assert ident.logical_device == f"colab:{card}"
        assert RemoteDeviceIdentity.from_dict(ident.to_dict()) == ident
        assert parse_remote_device(f"colab:{card}") == ("colab", card)

    with pytest.raises(DeviceRefused, match="provider"):
        RemoteDeviceIdentity(
            provider="aws",
            card="A100-40",
            vram_mib=40960,
            driver_version="550.54.15",
            cuda_version="12.4",
            pool="us-central1-a",
        )
    with pytest.raises(DeviceRefused, match="card"):
        RemoteDeviceIdentity(
            provider="colab",
            card="H100",
            vram_mib=81920,
            driver_version="550.54.15",
            cuda_version="12.4",
            pool="us-central1-a",
        )
    with pytest.raises(DeviceRefused, match="vram_mib"):
        RemoteDeviceIdentity(
            provider="colab",
            card="T4",
            vram_mib=0,
            driver_version="550.54.15",
            cuda_version="12.4",
            pool="us-central1-a",
        )
    with pytest.raises(DeviceRefused, match="driver_version"):
        RemoteDeviceIdentity(
            provider="colab",
            card="T4",
            vram_mib=15360,
            driver_version="",
            cuda_version="12.4",
            pool="us-central1-a",
        )
    with pytest.raises(DeviceRefused, match="cuda_version"):
        RemoteDeviceIdentity(
            provider="colab",
            card="T4",
            vram_mib=15360,
            driver_version="550.54.15",
            cuda_version="",
            pool="us-central1-a",
        )
    with pytest.raises(DeviceRefused, match="pool"):
        RemoteDeviceIdentity.from_dict({"provider": "colab"})
    with pytest.raises(DeviceRefused, match="mapping"):
        RemoteDeviceIdentity.from_dict("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(DeviceRefused, match="unhashed account id"):
        RemoteDeviceIdentity.from_dict(
            {
                "provider": "colab",
                "card": "T4",
                "vram_mib": 15360,
                "driver_version": "550.54.15",
                "cuda_version": "12.4",
                "pool": "us-central1-a",
                "account_id": "raw_acct",
            }
        )
    with pytest.raises(DeviceRefused, match="unexpected fields"):
        RemoteDeviceIdentity.from_dict(
            {
                "provider": "colab",
                "card": "T4",
                "vram_mib": 15360,
                "driver_version": "550.54.15",
                "cuda_version": "12.4",
                "pool": "us-central1-a",
                "extra_field": 1,
            }
        )
    with pytest.raises(DeviceRefused, match="not a remote device"):
        parse_remote_device("5090")
    with pytest.raises(DeviceRefused, match="unsupported remote provider"):
        parse_remote_device("aws:A100-40")
    with pytest.raises(DeviceRefused, match="unsupported colab card"):
        parse_remote_device("colab:H100")
    with pytest.raises(DeviceRefused, match="non-empty zone"):
        RemoteDeviceIdentity(
            provider="colab",
            card="T4",
            vram_mib=15360,
            driver_version="550.54.15",
            cuda_version="12.4",
            pool="",
        )


def test_remote_device_refuses_unhashed_account_in_pool():
    from interlock.devices import RemoteDeviceIdentity, validate_remote_pool

    assert validate_remote_pool("us-central1-a") == "us-central1-a"
    assert validate_remote_pool("europe-west4-b") == "europe-west4-b"
    hashed = "sha256:" + ("a" * 64)
    assert validate_remote_pool(hashed) == hashed

    for bad_pool in (
        "researcher@example.com",
        "account:researcher-01",
        "user:demo_acct",
        "uid_998877",
        "/" + "home/" + "researcher",
        "C:" + "\\Users\\" + "researcher",
    ):
        with pytest.raises(DeviceRefused, match="unhashed account id"):
            RemoteDeviceIdentity(
                provider="colab",
                card="L4",
                vram_mib=23034,
                driver_version="550.54.15",
                cuda_version="12.4",
                pool=bad_pool,
            )


@pytest.mark.parametrize(
    "field_name",
    ["provider", "card", "vram_mib", "driver_version", "cuda_version", "pool"],
)
@pytest.mark.parametrize(
    "account_input",
    [
        "researcher@example.com",
        "org/model-repo",
        "github-owner/gguf-weights",
        "github:octocat",
        "user:alice",
        "account:bob",
        "hf:org_handle",
        "/" + "home/" + "researcher",
        "C:" + "\\Users\\" + "researcher",
    ],
)
def test_remote_device_refuses_unhashed_account_across_all_six_fields(field_name, account_input):
    from interlock.devices import RemoteDeviceIdentity

    payload = {
        "provider": "colab",
        "card": "L4",
        "vram_mib": 23034,
        "driver_version": "550.54.15",
        "cuda_version": "12.4",
        "pool": "us-central1-a",
    }
    payload[field_name] = account_input
    with pytest.raises(DeviceRefused, match="unhashed account id"):
        RemoteDeviceIdentity.from_dict(payload)


def test_remote_device_requires_numeric_driver_and_cuda_versions():
    from interlock.devices import RemoteDeviceIdentity

    for bad_version in ("r550-beta", "latest", "550", "12", "v12.4", "12.4.1-rc1"):
        with pytest.raises(DeviceRefused, match="numeric version"):
            RemoteDeviceIdentity(
                provider="colab",
                card="L4",
                vram_mib=23034,
                driver_version=bad_version,
                cuda_version="12.4",
                pool="us-central1-a",
            )
        with pytest.raises(DeviceRefused, match="numeric version"):
            RemoteDeviceIdentity(
                provider="colab",
                card="L4",
                vram_mib=23034,
                driver_version="550.54.15",
                cuda_version=bad_version,
                pool="us-central1-a",
            )



def test_remote_and_rig_rows_stay_in_separate_pools():
    from interlock.devices import (
        RIG_DEVICES,
        assert_single_pool,
        device_pool_kind,
        is_colab_device,
        is_remote_device,
        is_rig_device,
        matches_device_filter,
        partition_rows_by_pool,
    )

    for rig in RIG_DEVICES:
        assert is_rig_device(rig) is True
        assert is_remote_device(rig) is False
        assert is_colab_device(rig) is False
        assert device_pool_kind(rig) == "rig"

    assert is_rig_device("colab:A100-40") is False
    assert is_remote_device("colab:A100-40") is True
    assert is_colab_device("colab:A100-40") is True
    assert device_pool_kind("colab:A100-40") == "remote"
    assert matches_device_filter("colab:A100-40", "colab:*") is True
    assert matches_device_filter("colab:T4", "colab:T4") is True
    assert matches_device_filter("5090", "colab:*") is False
    assert matches_device_filter("5090", None) is True
    assert matches_device_filter(None, "colab:*") is False
    assert is_remote_device(None) is False
    with pytest.raises(DeviceRefused, match="unknown device pool"):
        device_pool_kind("unknown:card")

    with pytest.raises(DeviceRefused):
        resolve_device("colab:A100-40", PRESENT)

    rig_row = {"id": "r1", "device": "5090"}
    remote_row = {"id": "c1", "device": "colab:A100-40"}
    pools = partition_rows_by_pool([rig_row, remote_row])
    assert pools["rig"] == [rig_row]
    assert pools["remote"] == [remote_row]
    assert assert_single_pool([rig_row]) == "rig"
    assert assert_single_pool([remote_row]) == "remote"
    with pytest.raises(DeviceRefused, match="never pooled"):
        assert_single_pool([rig_row, remote_row])

