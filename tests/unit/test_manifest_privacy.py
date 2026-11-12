import json
from pathlib import Path


def test_committed_sample_manifest_uses_repository_relative_paths():
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "data" / "processed" / "higgs_sample_250k.csv.gz.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in ("source_path", "output_path"):
        value = manifest.get(key)
        assert isinstance(value, str) and value.strip(), f"{key} must be present"
        assert not Path(value).is_absolute(), f"{key} must not expose a local absolute path"
        assert "Users/" not in value and "/content/" not in value
