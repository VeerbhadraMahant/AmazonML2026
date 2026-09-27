"""Build the final <team>_submission.zip in the layout required by the
challenge ("Final Submission Package"):

    <team>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/src/**      (no __pycache__ / *.pyc)
    ├── code/business_entity_resolution/README.md
    ├── code/business_entity_resolution/requirements.txt
    └── Documentation_template.md

Files are streamed into the archive with zf.write (ZIP_DEFLATED, level 6,
ZIP64 enabled), so memory stays low even for the ~1.6GB candidate_pairs.tsv.

Usage:
    python make_submission_zip.py --team NAME [--out PATH] [--dry-run]
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
CODE_DIR = HERE.parents[1]            # code/business_entity_resolution
ROOT = HERE.parents[3]                # repo root
OUTPUT_DIR = ROOT / "output"
DOC_PATH = ROOT / "AmazonML" / "student_resource" / "Documentation_template.md"

EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def collect_files() -> list[tuple[Path, str]]:
    """Returns (source_path, archive_name) pairs, in archive order."""
    items = [
        (OUTPUT_DIR / "matching_results.tsv", "output/matching_results.tsv"),
        (OUTPUT_DIR / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ]
    src_dir = CODE_DIR / "src"
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src_dir)
        if any(part in EXCLUDE_DIRS for part in rel.parts) or p.suffix in EXCLUDE_SUFFIXES:
            continue
        items.append((p, f"code/business_entity_resolution/src/{rel.as_posix()}"))
    items += [
        (CODE_DIR / "README.md", "code/business_entity_resolution/README.md"),
        (CODE_DIR / "requirements.txt", "code/business_entity_resolution/requirements.txt"),
        (DOC_PATH, "Documentation_template.md"),
    ]
    return items


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="team", help="team name (zip is <team>_submission.zip)")
    ap.add_argument("--out", default=None, help="output zip path (default: <repo>/<team>_submission.zip)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be included, write nothing")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / f"{args.team}_submission.zip"
    items = collect_files()

    missing = [str(p) for p, _ in items if not p.exists()]
    if missing:
        print("ERROR: missing required files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    total = sum(p.stat().st_size for p, _ in items)
    if args.dry_run:
        print(f"[dry-run] would write {out} with {len(items)} files, {fmt_size(total)} uncompressed:")
        for p, arc in items:
            print(f"  {fmt_size(p.stat().st_size):>10}  {arc}")
        return

    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zf:
        for p, arc in items:
            print(f"  adding {arc} ({fmt_size(p.stat().st_size)})", flush=True)
            zf.write(p, arc)

    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        infos = zf.infolist()
    print(f"\nwrote {out} in {round(time.time() - t0, 1)}s; "
          f"{len(infos)} files, {fmt_size(total)} -> {fmt_size(out.stat().st_size)}"
          f"; CRC check: {'OK' if bad is None else 'FAILED on ' + bad}")
    for zi in infos:
        print(f"  {fmt_size(zi.file_size):>10} -> {fmt_size(zi.compress_size):>10}  {zi.filename}")


if __name__ == "__main__":
    main()
