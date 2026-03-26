# Architecture diagrams (Graphviz)

This repo includes clean, editable architecture diagrams as Graphviz sources:

- `docs/architecture/acagn_training_pipeline.dot` — Diagram 1 (System Overview)
- `docs/architecture/acagn_inference.dot` — Diagram 2 (ACAGN-Gate Detail)

## Render to SVG/PNG (recommended: SVG)

1) Install Graphviz (one-time):
   - macOS (Homebrew): `brew install graphviz`
   - Debian/Ubuntu: `sudo apt-get install -y graphviz`
   - Windows (winget): `winget install Graphviz.Graphviz`

2) Render:

```bash
mkdir -p diagrams/architecture
bash scripts/render_arch_diagrams.sh svg
```

Optional PNG output:

```bash
bash scripts/render_arch_diagrams.sh png
```

## One-command render

If you prefer, run:

```bash
bash scripts/render_arch_diagrams.sh
```

Notes:
- Default output directory is `diagrams/architecture`.
