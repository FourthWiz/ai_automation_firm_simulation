#!/usr/bin/env python3.11
"""
Build a deterministic gzipped tarball of the firm_ai_abm embed kernel.

Usage:
    python scripts/bundle_kernel_for_embed.py [--site-root PATH] [--dry-run]

Output:
    <site_root>/public/sim/firm_ai_abm_embed.tar.gz
    <site_root>/public/sim/manifest.json

Design notes (T-06 / round-4 CRIT-1):
  - Pure-Python tarfile + gzip.GzipFile(mtime=0). No shell `tar` or `gzip` in
    the determinism path. Result: byte-identical archives on BSD-tar (macOS),
    GNU-tar (Linux), and any host with Python 3.11+.
  - Round-5 MIN-2: USTAR_FORMAT (not PAX). All bundled filenames are well under
    USTAR's 100-byte limit, so no PAX extended headers would fire anyway. USTAR
    has smaller cross-version surface area and is the de-facto reference format
    for deterministic Python tar cookbooks (Bazel pkg_tar default).
  - The only shell invocation is `git rev-parse HEAD` for the manifest's
    kernel_git_head field; it has no determinism impact.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, subprocess, sys, tarfile, time
from pathlib import Path

EXCLUDE_NAMES = {"__pycache__", ".DS_Store"}
EXCLUDE_PREFIXES = ("._",)   # iCloud AppleDouble shadows

# Narrow strip list verified at task time (round-2 MAJ-8):
# only these two files import incompatible modules at module scope.
STRIP_FILES = {
    "firm_ai_abm/viz.py",        # imports matplotlib at module scope
    "firm_ai_abm/validate.py",   # imports subprocess at module scope
}


def iter_kernel_files(repo_root: Path):
    """Yield (arcname, path) for every file to include in the tarball."""
    fa = repo_root / "firm_ai_abm"
    for p in sorted(fa.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root).as_posix()
        if any(part in EXCLUDE_NAMES for part in p.parts):
            continue
        if any(p.name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        if rel in STRIP_FILES:
            continue
        yield rel, p
    # Always include embed_app.py at archive root
    yield "embed_app.py", repo_root / "embed_app.py"


def kernel_content_hash(repo_root: Path) -> str:
    """SHA-256 of concatenated sorted *.py bytes — for staleness check (round-4 MIN-2)."""
    h = hashlib.sha256()
    fa = repo_root / "firm_ai_abm"
    for p in sorted(fa.rglob("*.py")):
        rel = p.relative_to(repo_root).as_posix()
        if any(part in EXCLUDE_NAMES for part in p.parts):
            continue
        if rel in STRIP_FILES:
            continue
        h.update(p.read_bytes())
    return h.hexdigest()


def build_deterministic_tarball(repo_root: Path) -> bytes:
    """Pure-Python deterministic tar.gz construction (no shell tar/gzip)."""
    buf = io.BytesIO()
    # mtime=0 zeroes the gzip header's embedded mtime.
    gz = gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0)
    # USTAR_FORMAT: all bundled filenames are well under the 100-byte limit;
    # smaller cross-version surface area than PAX_FORMAT.
    with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for arcname, path in iter_kernel_files(repo_root):
            data = path.read_bytes()
            ti = tarfile.TarInfo(name=arcname)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            ti.type = tarfile.REGTYPE
            tf.addfile(ti, io.BytesIO(data))
    gz.close()
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="Bundle firm_ai_abm embed kernel.")
    ap.add_argument(
        "--site-root", type=Path, required=False,
        default=Path(__file__).resolve().parents[2] / "Personal_Site" / "personal-site",
        help="Absolute path to the personal-site repo root.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned file list without writing anything.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]   # FirmBehavior/
    site_root = args.site_root.resolve()
    out_dir   = site_root / "public" / "sim"

    files = list(iter_kernel_files(repo_root))
    if args.dry_run:
        total_bytes = 0
        for arc, p in files:
            size = p.stat().st_size
            total_bytes += size
            print(f"  {arc:<52} ({size:>8} bytes)")
        print(f"total: {len(files)} files, {total_bytes} bytes")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = build_deterministic_tarball(repo_root)
    out_tar = out_dir / "firm_ai_abm_embed.tar.gz"
    out_tar.write_bytes(tarball)

    sha = hashlib.sha256(tarball).hexdigest()
    kernel_sha = kernel_content_hash(repo_root)
    kernel_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or "unknown"

    manifest = {
        "sha256": sha,
        "size_bytes": len(tarball),
        "kernel_git_head": kernel_head,
        "kernel_sha256": kernel_sha,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embed_app_version": "1.0.0",
        "stlite_pin": "@stlite/browser@1.7.3",
        "archive_format": "gztar",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {out_tar} ({len(tarball)} bytes, sha256={sha[:16]}…)")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    print(f"Next: cd {site_root!s} && git add public/sim/ && git commit -m 'sim: refresh embed kernel'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
