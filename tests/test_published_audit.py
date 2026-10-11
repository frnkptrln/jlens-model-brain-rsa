import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_confirmatory", ROOT / "scripts/audit_confirmatory.py"
)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


@pytest.fixture
def bundle(tmp_path):
    destination = tmp_path / "bundle"
    shutil.copytree(ROOT / "results/confirmatory", destination)
    return destination


def rewrite_csv(path, modify):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    modify(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_published_contrast_reproduces_without_rewriting_sources(bundle):
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}
    result = audit_module.audit(bundle)
    assert result["status"] == "pass"
    assert result["cell_count"] == 48
    assert result["sign_assignments"] == 256
    assert result["extreme_assignments"] == 2
    assert result["negative_participants"] == 8
    assert result["primary"]["p_signflip_two_sided"] == 0.0078125
    assert before == {path.name: path.read_bytes() for path in bundle.iterdir()}


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "delta", "nonfinite"])
def test_changed_or_incomplete_component_data_cannot_pass(bundle, mutation):
    def modify(rows):
        if mutation == "duplicate":
            rows.append(rows[0].copy())
        elif mutation == "missing":
            rows.pop()
        else:
            rows[0]["delta_jlens_minus_raw"] = "nan" if mutation == "nonfinite" else "0.0"

    rewrite_csv(bundle / "paired_contrasts.csv", modify)
    with pytest.raises(ValueError):
        audit_module.audit(bundle)


def test_stale_group_summary_is_detected(bundle):
    path = bundle / "primary_test.json"
    data = json.loads(path.read_text())
    data["p_signflip_two_sided"] = 0.05
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="p_signflip"):
        audit_module.audit(bundle)
