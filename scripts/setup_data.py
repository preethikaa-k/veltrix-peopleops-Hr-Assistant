"""Unpack the assessment zip into ./data so `docker compose up` and `make test` have real data.

    python scripts/setup_data.py path/to/veltrix_assessment.zip

Result:  data/corpus/*   data/manifest.json   data/veltrix.db   (data/ is git-ignored)
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
dest = Path(__file__).resolve().parent.parent / "data"
with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(tmp)
    root = next(p.parent for p in Path(tmp).rglob("manifest.json") if "__MACOSX" not in p.parts)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "corpus").mkdir(parents=True)
    for f in (root / "corpus").iterdir():
        shutil.copy(f, dest / "corpus" / f.name)
    shutil.copy(root / "manifest.json", dest / "manifest.json")
    shutil.copy(root / "veltrix.db", dest / "veltrix.db")
print(f"data ready in {dest}: {len(list((dest / 'corpus').iterdir()))} documents")
