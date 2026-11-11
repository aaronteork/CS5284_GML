"""Package code, all copied data, caches and guides; exclude local training artifacts."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def included(path):
    relative = path.relative_to(ROOT)
    if any(part in {"__pycache__", ".git", ".DS_Store", ".ipynb_checkpoints"} for part in relative.parts):
        return False
    if relative.parts[0] in {"experiments", "weights", "results"}:
        return False
    if any(part.startswith("local_smoke") for part in relative.parts):
        return False
    return path.is_file() and path.suffix != ".pyc"


def main():
    archive = ROOT.parent / "train_historydata_colab.zip"
    sources = [p for p in ROOT.rglob("*") if included(p) and p.name != "RELEASE_MANIFEST.json"]
    code = [p for p in sources if p.relative_to(ROOT).parts[0] in {"src", "scripts", "notebooks", "tests"}
            or p.name in {"requirements.txt", "README.md", "COLAB_GUIDE.md", "DATA_AND_INTEGRATION.md"}]
    manifest = {"code_and_guides": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(code)},
                "original_data_manifest": "DATA_COPY_MANIFEST.json",
                "data_files": 379, "training_outputs_included": False}
    (ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sources.append(ROOT / "RELEASE_MANIFEST.json")
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(sources):
            z.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(temporary) as z:
        error = z.testzip()
        if error:
            raise RuntimeError(f"ZIP integrity check failed: {error}")
        names = set(z.namelist())
        for item in json.loads((ROOT / "DATA_COPY_MANIFEST.json").read_text())["files"]:
            assert str(Path(ROOT.name) / item["path"]) in names
    temporary.replace(archive)
    print(json.dumps({"archive": str(archive), "files": len(sources),
                      "compressed_mib": archive.stat().st_size / 1024**2,
                      "uncompressed_mib": sum(p.stat().st_size for p in sources) / 1024**2,
                      "zip_integrity": "passed", "all_379_original_data_files_included": True}, indent=2))


if __name__ == "__main__":
    main()
