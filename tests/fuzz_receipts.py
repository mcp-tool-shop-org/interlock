"""Coverage-guided fuzz target for a receipt line and an experiment spec.

Each input is split by atheris and written under a temporary root. One half
goes to ``table/routing.jsonl`` (the line ``interlock check`` reads through
``check_tree``) and to ``receipts/row.json`` (the document reader on that same
path; a ``*.run.json`` marker is not a receipt). The other half goes to a
spec file, ``.json`` or ``.yaml``, and ``load_spec`` reads it.

A refusal is the expected outcome. Only an exception outside that path fails
the run. This module is named ``fuzz_*`` so pytest does not collect it, and
it imports atheris only when run as a program.
"""

from __future__ import annotations

import sys

_MAX_BYTES = 8192


def main() -> None:
    import atheris

    with atheris.instrument_imports():
        import json
        import tempfile
        from pathlib import Path

        import yaml

        from interlock.harness.spec import SpecError, load_spec
        from interlock.receipt import ReceiptError, check_tree

    def refuse(read, kinds: tuple[type[BaseException], ...]) -> None:
        try:
            read()
        except kinds:
            return

    def test_one_input(data: bytes) -> None:
        capped = data[:_MAX_BYTES]
        provider = atheris.FuzzedDataProvider(capped)
        as_json = provider.ConsumeBool()
        receipt = provider.ConsumeBytes(len(capped) // 2)
        spec_bytes = provider.ConsumeBytes(len(capped))
        suffix = ".json" if as_json else ".yaml"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "receipts").mkdir()
            (root / "table").mkdir()
            (root / "table" / "routing.jsonl").write_bytes(receipt)
            (root / "receipts" / "row.json").write_bytes(receipt)
            refuse(
                lambda: check_tree(root),
                (ReceiptError, json.JSONDecodeError, UnicodeDecodeError),
            )
            spec_path = root / f"spec{suffix}"
            spec_path.write_bytes(spec_bytes)
            refuse(
                lambda: load_spec(spec_path),
                (SpecError, json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError),
            )

    atheris.Setup(sys.argv, atheris.instrument_func(test_one_input))
    atheris.Fuzz()


if __name__ == "__main__":
    main()
