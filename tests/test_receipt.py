import json
from pathlib import Path

import pytest

from interlock.receipt import ReceiptError, check_tree, validate_row
from tests.support import row


def test_a_complete_row_validates():
    validate_row(row())


def test_schema_refuses_gpu_1_and_a_broken_interval():
    bad_device = row(device="GPU.1")
    with pytest.raises(ReceiptError, match="device"):
        validate_row(bad_device)
    bad_interval = row()
    bad_interval["quality"][0]["interval"] = {"low": 0.9, "high": 0.1, "method": "wilson-95"}
    with pytest.raises(ReceiptError, match="interval"):
        validate_row(bad_interval)


def test_home_directory_paths_and_unpinned_bytes_are_refused():
    homed = row()
    # Built in pieces so the test source itself is not a home-directory path.
    homed["notes"] = "C:" + "\\Users\\" + "someone\\receipt.json"
    with pytest.raises(ReceiptError, match="home-directory"):
        validate_row(homed)
    unpinned = row(provenance={"source_receipt_path": "E:/AI/rnd-ladder/ladder-qwen3_8b/metrics.json",
        "commit_sha": None})
    with pytest.raises(ReceiptError, match="content_sha256"):
        validate_row(unpinned)
    unpinned["provenance"]["content_sha256"] = "deadbeef"
    validate_row(unpinned)


def test_device_lost_cannot_carry_timing():
    lost = row()
    lost["outcome"] = "device_lost"
    lost["latency"] = {
        "cold_ms": None,
        "warm_p50_ms": None,
        "warm_p95_ms": None,
        "throughput": None,
        "throughput_unit": None,
        "compile_ms": None,
        "first_call_ms": None,
    }
    lost["quality"] = []
    validate_row(lost)
    lost["latency"]["cold_ms"] = 12
    with pytest.raises(ReceiptError, match="no timing"):
        validate_row(lost)
    lost["latency"]["cold_ms"] = None
    lost["artifacts"] = {"slots": [{"latency_ms": 12}]}
    with pytest.raises(ReceiptError, match="no timing"):
        validate_row(lost)
    hung = row()
    hung["outcome"] = "hang"
    hung["latency"]["cold_ms"] = 60_000
    with pytest.raises(ReceiptError, match="a hang row has no timing"):
        validate_row(hung)
    lost.pop("artifacts", None)
    lost["latency"]["cold_ms"] = None
    lost["latency"]["compile_warm_reload_ms"] = 12
    with pytest.raises(ReceiptError, match="no timing"):
        validate_row(lost)


def test_a_failed_parity_guard_keeps_its_timing():
    kept = row()
    kept["parity_valid"] = False
    kept["latency"]["compile_warm_reload_ms"] = 12
    kept["settings"]["prompt_tokens"] = 2048
    kept["settings"]["truncated_tokens"] = 1920
    validate_row(kept)


def test_experiment_provenance_is_the_other_legal_shape():
    validate_row(row(provenance={"experiment_spec": "experiments/e1-cold-warm/spec.yaml", "run_id": "dry-run"}))


def test_check_is_green_on_an_empty_tree_and_red_on_a_bad_receipt(tmp_path: Path):
    assert check_tree(tmp_path) == []
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "ok.json").write_text(json.dumps(row()), encoding="utf-8")
    assert check_tree(tmp_path) == []
    (receipts / "bad.json").write_text("{", encoding="utf-8")
    problems = check_tree(tmp_path)
    assert len(problems) == 1
    assert "bad.json" in problems[0]


def test_a_receipt_that_is_not_utf8_is_a_problem(tmp_path: Path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "row.json").write_bytes(b"\xff\xfe{")
    problems = check_tree(tmp_path)
    assert len(problems) == 1
    assert "row.json" in problems[0]
    assert "utf-8" in problems[0]


def test_check_reads_a_jsonl_table(tmp_path: Path):
    table = tmp_path / "table"
    table.mkdir()
    (table / "routing.jsonl").write_text(json.dumps(row()) + "\n" + json.dumps(row(id="fixture/two")) + "\n",
        encoding="utf-8")
    assert check_tree(tmp_path) == []
    (table / "routing.jsonl").write_text("{not json}\n", encoding="utf-8")
    assert check_tree(tmp_path)


def test_a_list_of_rows_and_a_non_object_are_both_handled(tmp_path: Path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "many.json").write_text(json.dumps([row(), row(id="fixture/two")]), encoding="utf-8")
    assert check_tree(tmp_path) == []
    (receipts / "nope.json").write_text("1", encoding="utf-8")
    assert any("nope.json" in problem for problem in check_tree(tmp_path))


ROOT = Path(__file__).resolve().parents[1]


def test_privacy_check_refuses_home_paths_hostnames_and_token_shapes():
    for planted in ("/home/" + "someone/receipt.json", "/Users/" + "someone/receipt.json"):
        homed = row()
        homed["notes"] = planted
        with pytest.raises(ReceiptError, match="home-directory"):
            validate_row(homed)

    remote = row()
    remote["notes"] = "10.1.2.3"
    with pytest.raises(ReceiptError, match="hostname"):
        validate_row(remote)
    named = row()
    named["notes"] = "http://gpu-box.local/api"
    with pytest.raises(ReceiptError, match="hostname"):
        validate_row(named)

    for prefix in ("hf" + "_", "sk" + "-", "ghp" + "_"):
        token = row()
        token["notes"] = prefix + "example"
        with pytest.raises(ReceiptError, match="token-shaped"):
            validate_row(token)

    hex_secret = row()
    hex_secret["settings"]["api_token"] = "0123456789abcdef" * 3
    with pytest.raises(ReceiptError, match="token-shaped"):
        validate_row(hex_secret)
    b64_secret = row()
    b64_secret["settings"]["client_secret"] = ("ab" * 16) + "+/=="
    with pytest.raises(ReceiptError, match="token-shaped"):
        validate_row(b64_secret)


def test_privacy_check_allows_loopback_a_commit_sha_and_a_filename():
    allowed = row()
    allowed["notes"] = "http://127.0.0.1:11434 and localhost. See tail-c.log. The figure is cost.seconds."
    allowed["notes"] += "0123456789abcdef" * 4
    validate_row(allowed)
    assert len(row()["provenance"]["commit_sha"]) >= 32
    validate_row(row())


def test_check_reports_a_privacy_hit_in_a_receipt(tmp_path: Path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    planted = row()
    planted["notes"] = "10.1.2.3"
    (receipts / "planted.json").write_text(json.dumps(planted), encoding="utf-8")
    problems = check_tree(tmp_path)
    assert len(problems) == 1
    assert "hostname" in problems[0]
    assert "planted.json" in problems[0]


def test_committed_receipts_and_table_rows_pass_the_privacy_check():
    assert check_tree(ROOT) == []


def test_privacy_check_refuses_unc_paths():
    planted = row()
    planted["notes"] = "\\\\" + "server\\share\\file"
    with pytest.raises(ReceiptError, match="UNC path"):
        validate_row(planted)


def test_privacy_check_refuses_non_http_schemes_and_user_at_host():
    for planted in (
        "ssh" + "://myrig/repo",
        "sftp" + "://myrig/repo",
        "ssh" + "://gpu-box.local/repo",
        "git" + "@" + "myrig:org/repo",
        "git" + "@" + "example.com:org/repo",
        "ada" + "@" + "box.corp",
    ):
        remote = row()
        remote["notes"] = planted
        with pytest.raises(ReceiptError, match="hostname"):
            validate_row(remote)
    allowed = row()
    allowed["notes"] = "ssh" + "://127.0.0.1/repo and git" + "@" + "localhost:repo"
    validate_row(allowed)


def test_privacy_check_refuses_more_token_prefixes():
    for prefix in (
        "github" + "_pat_",
        "gh" + "o_",
        "gh" + "s_",
        "gh" + "u_",
        "xox" + "a-",
        "xox" + "b-",
        "xox" + "p-",
        "xox" + "r-",
    ):
        token = row()
        token["notes"] = prefix + "example"
        with pytest.raises(ReceiptError, match="token-shaped"):
            validate_row(token)
    access_key = row()
    access_key["notes"] = "AKIA" + ("A1" * 8)
    with pytest.raises(ReceiptError, match="token-shaped"):
        validate_row(access_key)
    bearer = row()
    bearer["notes"] = "Bearer " + ("a" * 16)
    with pytest.raises(ReceiptError, match="token-shaped"):
        validate_row(bearer)


def test_an_expanded_userprofile_path_is_already_refused():
    for planted in (
        "C:" + "\\Users\\" + "someone\\receipt.json",
        "C:" + "/Users/" + "someone/receipt.json",
    ):
        homed = row()
        homed["notes"] = planted
        with pytest.raises(ReceiptError, match="home-directory"):
            validate_row(homed)


def test_a_bare_host_port_is_a_known_limit_until_host_port_is_set():
    bare = row()
    bare["notes"] = "myrig:11434"
    validate_row(bare)
    with pytest.raises(ReceiptError, match="hostname"):
        validate_row(bare, host_port=True)
    tagged = row()
    tagged["notes"] = "gemma4:31b and localhost:11434 and 127.0.0.1:11434 and myrig:65536"
    validate_row(tagged, host_port=True)


def test_committed_receipts_pass_the_host_port_check():
    assert check_tree(ROOT, host_port=True) == []


def _remote_row(**overrides):
    card = overrides.pop("card", "A100-40")
    remote_device = overrides.pop(
        "remote_device",
        {
            "provider": "colab",
            "card": card,
            "vram_mib": 40960,
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "pool": "us-central1-a",
        },
    )
    base = row(
        id=f"C2-GENERATE/qwen3:8b/colab-{card}",
        task="generate",
        engine={"name": "ollama", "version": "0.35.1"},
        device=f"colab:{card}",
        remote_device=remote_device,
        provenance={"experiment_spec": "experiments/c2-generate/spec.yaml", "run_id": "c2-win-1"},
    )
    base.update(overrides)
    return base


def test_remote_colab_rows_and_failure_outcomes_validate():
    for card in ("A100-40", "A100-80", "L4", "T4", "G4"):
        valid = _remote_row(card=card)
        validate_row(valid)

    for outcome, note in (
        ("oom", "oom: CUDA out of memory at num_ctx 32768 (peak 15320 MiB / 15360 MiB)"),
        ("load_refused", "load_refused: model requires 32000 MiB, colab:T4 has 15360 MiB"),
        ("timeout", "timeout: request exceeded 120s deadline at concurrency 8"),
        ("ctx_exceeded", "ctx_exceeded: num_ctx 65536 exceeds max_resident_ctx 16384"),
    ):
        failed = _remote_row(
            card="T4",
            outcome=outcome,
            notes=note,
            quality=[],
            latency={
                "cold_ms": None,
                "warm_p50_ms": None,
                "warm_p95_ms": None,
                "throughput": None,
                "throughput_unit": None,
                "compile_ms": None,
            },
        )
        validate_row(failed)


def test_remote_row_refuses_missing_identity_mismatched_card_rig_mixing_or_unhashed_account():
    unhashed_rig_setting = row(
        settings={**row()["settings"], "account_id": "raw_account_handle", "tags": [{"user_id": "u1"}]}
    )
    with pytest.raises(ReceiptError, match="unhashed account id"):
        validate_row(unhashed_rig_setting)

    mixed_rig = row(
        device="5090",
        engine={"name": "ollama", "version": "0.35.1"},
        remote_device={
            "provider": "colab",
            "card": "A100-40",
            "vram_mib": 40960,
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "pool": "us-central1-a",
        },
    )
    with pytest.raises(ReceiptError, match="never pooled"):
        validate_row(mixed_rig)

    missing = _remote_row()
    del missing["remote_device"]
    with pytest.raises(ReceiptError, match="remote_device"):
        validate_row(missing)

    mismatched = _remote_row(
        card="A100-40",
        remote_device={
            "provider": "colab",
            "card": "T4",
            "vram_mib": 15360,
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "pool": "us-central1-a",
        },
    )
    with pytest.raises(ReceiptError, match="does not match"):
        validate_row(mismatched)

    unhashed_pool = _remote_row(
        remote_device={
            "provider": "colab",
            "card": "A100-40",
            "vram_mib": 40960,
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "pool": "researcher@example.com",
        }
    )
    with pytest.raises(ReceiptError, match="unhashed account id"):
        validate_row(unhashed_pool)

    unhashed_setting = _remote_row(
        settings={**row()["settings"], "account_id": "raw_account_handle", "tags": [{"user_id": "u1"}]}
    )
    with pytest.raises(ReceiptError, match="unhashed account id"):
        validate_row(unhashed_setting)

    hashed_setting = _remote_row(
        settings={**row()["settings"], "account_id": "sha256:" + ("b" * 64), "tags": [{"note": "ok"}]}
    )
    validate_row(hashed_setting)


def test_validate_row_refuses_unhashed_account_in_any_remote_device_field_or_gguf_repo():
    for field_name in ("provider", "card", "vram_mib", "driver_version", "cuda_version", "pool"):
        for planted in (
            "researcher@example.com",
            "org/model-repo",
            "user:alice",
            "github:octocat",
            "/" + "home/" + "researcher",
            "C:" + "\\Users\\" + "researcher",
        ):
            bad_remote = {
                "provider": "colab",
                "card": "A100-40",
                "vram_mib": 40960,
                "driver_version": "550.54.15",
                "cuda_version": "12.4",
                "pool": "us-central1-a",
            }
            bad_remote[field_name] = planted
            candidate = _remote_row(remote_device=bad_remote)
            with pytest.raises(ReceiptError, match="unhashed account id"):
                validate_row(candidate)

    for gguf_key in ("ggufRepo", "gguf_repo", "gguf_source"):
        unhashed_gguf = _remote_row(settings={**row()["settings"], gguf_key: "owner/role-set-gguf"})
        with pytest.raises(ReceiptError, match="unhashed account id"):
            validate_row(unhashed_gguf)
        hashed_gguf = _remote_row(settings={**row()["settings"], gguf_key: "sha256:" + ("c" * 64)})
        validate_row(hashed_gguf)


@pytest.mark.parametrize(
    ("case_name", "overrides"),
    [
        ("notes_account_prefix", {"notes": "account:some_handle ran this"}),
        (
            "settings_failure_reason_owner_prefix",
            {"settings": {**row()["settings"], "failure_reason": "owner:some_handle"}},
        ),
        (
            "settings_nested_list_account_prefix",
            {"settings": {**row()["settings"], "tags": ["ok", {"detail": "user:some_handle"}]}},
        ),
        (
            "top_level_string_account_prefix",
            {"id": "account:some_handle/qwen3:8b/colab-A100-40"},
        ),
    ],
)
def test_remote_row_refuses_account_prefix_in_any_string_field(case_name: str, overrides: dict):
    candidate = _remote_row(**overrides)
    with pytest.raises(ReceiptError, match="unhashed account id"):
        validate_row(candidate)


def test_remote_row_allows_hashed_identifiers_where_permitted():
    hashed_id = "sha256:" + ("d" * 64)
    allowed = _remote_row(
        remote_device={
            "provider": "colab",
            "card": "A100-40",
            "vram_mib": 40960,
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "pool": hashed_id,
        },
        notes=f"measured role-set {hashed_id}",
        settings={
            **row()["settings"],
            "account": hashed_id,
            "account_id": hashed_id,
            "user_id": hashed_id,
            "owner_id": hashed_id,
            "gguf_source": hashed_id,
            "tags": [hashed_id, {"digest": hashed_id}],
        },
    )
    validate_row(allowed)

    rig_candidate = row(
        notes="(account:some_handle) user-facing latency is fine",
        settings={
            **row()["settings"],
            "account:some_handle": "x",
            "meta": {"owner-some_handle": 1},
            "tags": [{"GitHub_some_handle": True}],
        },
    )
    validate_row(rig_candidate)


@pytest.mark.parametrize(
    ("case_name", "overrides", "must_refuse"),
    [
        ("val_paren_account", {"notes": "(account:some_handle)"}, True),
        ("val_comma_account", {"notes": "ran,account:some_handle"}, True),
        ("val_equals_owner", {"notes": "by=owner:some_handle"}, True),
        ("val_quoted_user", {"notes": '"user:some_handle"'}, True),
        ("val_semicolon_uid", {"notes": "x;uid:42"}, True),
        ("val_account_equals", {"notes": "account=some_handle"}, True),
        ("val_login_colon", {"notes": "login:some_handle"}, True),
        ("val_handle_equals", {"notes": "handle=some_handle"}, True),
        (
            "key_top_account_colon",
            {"settings": {**row()["settings"], "account:some_handle": "x"}},
            True,
        ),
        (
            "key_nested_owner_hyphen",
            {"settings": {**row()["settings"], "meta": {"owner-some_handle": 1}}},
            True,
        ),
        (
            "key_list_github_underscore",
            {"settings": {**row()["settings"], "tags": [{"GitHub_some_handle": True}]}},
            True,
        ),
        ("pass_user_facing_latency", {"notes": "user-facing latency is fine"}, False),
        ("pass_hf_transfer", {"notes": "hf-transfer was enabled"}, False),
        ("pass_github_hosted", {"notes": "github-hosted runner"}, False),
        ("pass_owner_of_the_run", {"notes": "owner of the run"}, False),
        ("pass_owner_ok_check", {"notes": "owner_ok check"}, False),
        ("pass_user_prompt_length", {"notes": "user_prompt length 512"}, False),
        ("pass_uid_less_session", {"notes": "uid-less session"}, False),
    ],
)
def test_remote_row_precise_account_rule_values_and_keys(
    case_name: str, overrides: dict, must_refuse: bool
):
    candidate = _remote_row(**overrides)
    if must_refuse:
        with pytest.raises(ReceiptError, match="unhashed account id"):
            validate_row(candidate)
    else:
        validate_row(candidate)

