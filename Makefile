.PHONY: check-embed

check-embed:
	.venv/bin/python scripts/check_embed_bundle_freshness.py
