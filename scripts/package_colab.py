"""Package code, all copied data, caches and guides; exclude local training artifacts."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path("train_historydata")


def included(path):
    relative = path.relative_to(ROOT)
    if any(part in {"__pycache__", ".git", ".DS_Store", ".ipynb_checkpoints"} for part in relative.parts):
        return False
    if relative.parts[0] not in {"src", "scripts", "notebooks", "tests", "docs", "data"} and not (
        len(relative.parts) == 1 and path.name in {"requirements.txt", "README.md", "Final_Report.ipynb"}
    ):
        return False
    if any(part.startswith("local_smoke") for part in relative.parts):
        return False
    return path.is_file() and path.suffix != ".pyc"


def main():
    archive = ROOT / "train_historydata_colab.zip"
    sources = [p for p in ROOT.rglob("*") if included(p) and p.name != "RELEASE_MANIFEST.json"]
    code = [p for p in sources if p.relative_to(ROOT).parts[0] != "data"]
    manifest = {"code_and_guides": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(code)},
                "data_files": sum(p.relative_to(ROOT).parts[0] == "data" for p in sources),
                "training_outputs_included": False}
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(sources):
            z.write(path, PACKAGE_ROOT / path.relative_to(ROOT))
        z.writestr((PACKAGE_ROOT / "RELEASE_MANIFEST.json").as_posix(), json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(temporary) as z:
        error = z.testzip()
        if error:
            raise RuntimeError(f"ZIP integrity check failed: {error}")
        names = set(z.namelist())
        for path in sources:
            if (PACKAGE_ROOT / path.relative_to(ROOT)).as_posix() not in names:
                raise RuntimeError(f"Missing packaged input: {path}")
    temporary.replace(archive)
    print(json.dumps({"archive": str(archive), "files": len(sources) + 1,
                      "compressed_mib": archive.stat().st_size / 1024**2,
                      "uncompressed_mib": sum(p.stat().st_size for p in sources) / 1024**2,
                      "zip_integrity": "passed", "data_files": manifest["data_files"]}, indent=2))


if __name__ == "__main__":
    main()
