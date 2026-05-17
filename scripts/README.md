# scripts/

## bundle_kernel_for_embed.py

Builds `personal-site/public/sim/firm_ai_abm_embed.tar.gz` — a deterministic
gzipped tarball of the `firm_ai_abm/` package plus `embed_app.py` for the
in-browser stlite embed at `igorban.ai/sim`.

Run after any change to `firm_ai_abm/*.py` or `embed_app.py`:

```
.venv/bin/python scripts/bundle_kernel_for_embed.py
```

## check_embed_bundle_freshness.py

Advisory check: warns if the tarball in `personal-site/public/sim/` is stale
relative to the current kernel source. Always exits 0. Run via `make check-embed`.
