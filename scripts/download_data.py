"""
Download NSL-KDD into ./data.

NSL-KDD is hosted at the University of New Brunswick. Mirrors come and go,
so this script tries a list of known URLs in order.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Try these mirrors in order. If all fail, the user must place the files manually.
MIRRORS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
    ],
}


def _download(name: str, urls: list[str], dst: Path) -> bool:
    for url in urls:
        try:
            print(f"  trying {url}")
            urlretrieve(url, dst)
            print(f"  saved -> {dst}")
            return True
        except (URLError, OSError) as e:
            print(f"  failed: {e}")
    return False


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NSL-KDD into {DATA_DIR}")

    success = True
    for name, urls in MIRRORS.items():
        target = DATA_DIR / name
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[fetch] {name}")
        if not _download(name, urls, target):
            success = False
            print(
                f"[error] could not download {name}. Place it manually at {target}\n"
                f"       Source: https://www.unb.ca/cic/datasets/nsl.html"
            )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
