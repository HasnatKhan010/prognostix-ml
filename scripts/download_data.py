#!/usr/bin/env python
"""Fetch the CMAPSS turbofan degradation dataset into ``data/raw/CMAPSS``.

The dataset is a public NASA release, but its hosting has moved several times, so
a list of candidate mirrors is tried in order. When every mirror fails the script
prints manual instructions instead of failing silently - a wrong or truncated
archive is worse than no archive.

Usage::

    python scripts/download_data.py                  # fetch if missing
    python scripts/download_data.py --force          # re-download
    python scripts/download_data.py --source ~/CMAPSSData.zip   # use a local zip
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, setup_logging  # noqa: E402

logger = logging.getLogger("prognostix.download")

MIRRORS = (
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
)

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED = tuple(
    f"{prefix}_{subset}.txt"
    for subset in SUBSETS
    for prefix in ("train", "test", "RUL")
)


def missing_files(directory: Path) -> list[str]:
    """Expected dataset files that are absent from ``directory``."""
    return [name for name in EXPECTED if not (directory / name).exists()]


def download(url: str, destination: Path, timeout: int = 120) -> Path:
    """Stream a URL to disk, reporting progress."""
    import requests

    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0

        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total:
                    percent = written / total * 100
                    print(
                        f"\r  {written / 1e6:7.1f} / {total / 1e6:.1f} MB ({percent:5.1f}%)",
                        end="",
                        flush=True,
                    )
    if total:
        print()
    logger.info("Downloaded %.1f MB", written / 1e6)
    return destination


def extract(archive: Path, target: Path, depth: int = 0) -> int:
    """Extract ``*.txt``/``*.pdf`` members, recursing into nested archives.

    The NASA bundle wraps ``CMAPSSData.zip`` inside an outer zip, so one pass is
    not enough. Members are written flat into ``target`` and paths are checked to
    keep a crafted archive from escaping it.
    """
    if depth > 3:
        logger.warning("Stopping at nesting depth %d in %s", depth, archive.name)
        return 0

    target.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            suffix = Path(name).suffix.lower()

            if suffix == ".zip":
                with tempfile.TemporaryDirectory() as staging:
                    nested = Path(staging) / name
                    with bundle.open(member) as source, nested.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    extracted += extract(nested, target, depth + 1)
                continue

            if suffix not in {".txt", ".pdf"}:
                continue

            destination = (target / name).resolve()
            if not str(destination).startswith(str(target.resolve())):
                logger.warning("Skipping unsafe member path: %s", member.filename)
                continue
            with bundle.open(member) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1

    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist."
    )
    parser.add_argument("--url", default=None, help="Download from this URL only.")
    parser.add_argument(
        "--source", default=None, help="Extract from an already-downloaded local zip."
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout.")
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()
    target = config.path("data_raw")
    target.mkdir(parents=True, exist_ok=True)

    absent = missing_files(target)
    if not absent and not args.force:
        logger.info("Dataset already present in %s - nothing to do", target)
        return 0
    if absent:
        logger.info("Missing %d file(s), e.g. %s", len(absent), ", ".join(absent[:4]))

    if args.source:
        source = Path(args.source).expanduser()
        if not source.exists():
            logger.error("%s not found", source)
            return 1
        count = extract(source, target)
        logger.info("Extracted %d file(s) from %s", count, source.name)
        return _verify(target)

    urls = [args.url] if args.url else list(MIRRORS)
    with tempfile.TemporaryDirectory() as staging:
        archive = Path(staging) / "cmapss.zip"
        for url in urls:
            try:
                download(url, archive, args.timeout)
                count = extract(archive, target)
                logger.info("Extracted %d file(s)", count)
                if not missing_files(target):
                    return _verify(target)
                logger.warning("Archive from %s did not contain the expected files", url)
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                logger.warning("Mirror failed (%s): %s", url.split("/")[2], exc)

    _print_manual_instructions(target)
    return 1


def _verify(target: Path) -> int:
    """Confirm the expected files landed and are non-trivial in size."""
    absent = missing_files(target)
    if absent:
        logger.error("Still missing %d file(s): %s", len(absent), ", ".join(absent[:6]))
        return 1

    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        size = (target / name).stat().st_size
        if size < 1024:
            logger.error("%s looks truncated (%d bytes)", name, size)
            return 1

    logger.info("Dataset verified in %s (%d files)", target, len(EXPECTED))
    print("\nNext step: python scripts/prepare_data.py")
    return 0


def _print_manual_instructions(target: Path) -> None:
    print(
        "\n".join(
            [
                "",
                "Automatic download failed. To install the dataset manually:",
                "",
                "  1. Download the 'Turbofan Engine Degradation Simulation Data Set'",
                "     from the NASA Prognostics Data Repository.",
                f"  2. Unzip it and copy the *.txt files into:  {target}",
                "  3. Re-run this script to verify, or go straight to:",
                "         python scripts/prepare_data.py",
                "",
                "Expected files: train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt (X = 1..4)",
                "",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
