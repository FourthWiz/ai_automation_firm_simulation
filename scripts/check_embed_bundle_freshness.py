#!/usr/bin/env python3
"""
Advisory staleness check: is the embedded kernel tarball up to date?

Compares the current kernel content hash to the one recorded in
personal-site/public/sim/manifest.json. Exits 0 in all cases (advisory only;
never blocks a build or commit).

Keyed on kernel_sha256 (content hash of all firm_ai_abm/*.py) so iCloud Drive
mtime churn does not produce false positives (round-4 MIN-2).

Usage:
    python scripts/check_embed_bundle_freshness.py [--site-root PATH]
    make check-embed
"""
from __future__ import annotations
import json, sys
from pathlib import Path

# Import the same hash function used by the bundler (keeps them in lockstep).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle_kernel_for_embed import kernel_content_hash  # noqa: E402


_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_RESET  = "\033[0m"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Check embed kernel bundle freshness.")
    ap.add_argument(
        "--site-root", type=Path, required=False,
        default=Path(__file__).resolve().parents[2] / "Personal_Site" / "personal-site",
    )
    args = ap.parse_args()

    repo_root  = Path(__file__).resolve().parents[1]
    site_root  = args.site_root.resolve()
    manifest_p = site_root / "public" / "sim" / "manifest.json"

    if not manifest_p.exists():
        print(f"{_YELLOW}WARN: manifest not found at {manifest_p}. "
              f"Run scripts/bundle_kernel_for_embed.py first.{_RESET}")
        return 0

    manifest = json.loads(manifest_p.read_text())
    recorded_hash = manifest.get("kernel_sha256", "")
    current_hash  = kernel_content_hash(repo_root)

    if current_hash == recorded_hash:
        print(f"{_GREEN}kernel_sha256 match: tarball is up to date ({current_hash[:16]}…){_RESET}")
    else:
        print(f"{_YELLOW}kernel_sha256 drift detected:{_RESET}")
        print(f"  manifest: {recorded_hash[:16]}…")
        print(f"  current:  {current_hash[:16]}…")
        print(f"  Run: cd {repo_root!s} && .venv/bin/python scripts/bundle_kernel_for_embed.py")
        print(f"       cd {site_root!s} && git add public/sim/ && git commit -m 'sim: refresh embed kernel'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
