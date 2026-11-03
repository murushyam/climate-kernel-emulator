"""Download ClimateBench v1.0.0 (Watson-Parris et al. 2022) from Zenodo record 7064308.

    python scripts/download_climatebench.py --out data/climatebench [--dry-run] [--keep-archives]
"""
import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

RECORD = "7064308"  # https://doi.org/10.5281/zenodo.7064308
FILES = {
    "train_val.tar.gz": "9bf0ebd9662667cc1ed78fd3eb5cad4e",  # ~839 MB
    "test.tar.gz": "a09a3ddc8c24e1d6d4b1ee4be5936510",       # ~74 MB
}
URL = "https://zenodo.org/records/{rec}/files/{name}?download=1"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/climatebench")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    for name, digest in FILES.items():
        url = URL.format(rec=RECORD, name=name)
        dest = out / name
        marker = out / f".{name}.extracted"
        if a.dry_run:
            print(f"[dry-run] {url} -> {dest} (md5 {digest})")
            continue
        out.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            print(f"skip {name}: already extracted")
            continue
        if not (dest.exists() and md5(dest) == digest):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        got = md5(dest)
        if got != digest:
            raise SystemExit(f"checksum mismatch for {name}: {got} != {digest}")
        with tarfile.open(dest) as t:
            t.extractall(out)
        marker.touch()
        if not a.keep_archives:
            dest.unlink()
        print(f"ok {name}")
    print("Note: archives may extract into a subfolder; point data.root at the folder holding inputs_*.nc.")


if __name__ == "__main__":
    main()
