# AsyncWebRL Documentation

Source for the AsyncWebRL documentation site (Sphinx + MyST + the Read the Docs
theme).

## Building locally

```bash
# Install doc dependencies once
pip install -r requirements-docs.txt

# Build and serve
./host.sh             # builds, then python -m http.server on :8000
#   http://localhost:8000/        English

# Or just build (no server)
./build.sh            # -> _build/html
```

## Layout

- `index.md` — landing page + toctree
- `intro.md` — project intro
- `installation.md` / `quickstart.md` — getting started (installation includes
  the verified H100 sm90 reference setup as an example)
- `async_system.md` / `algorithm.md` — paper-aligned design docs
- `cli_reference.md` — hand-curated reference of the config groups that matter
  for AsyncWebRL (not the full AReaL config surface)
- `conf.py` — Sphinx config
- `../.readthedocs.yaml` — Read the Docs build config
- `_static/custom.css` — theme tweaks

For the complete AReaL framework reference (FSDP/Megatron engines, SGLang/vLLM
inference, alloc modes, checkpoint formats, every config dataclass), see the
[upstream AReaL documentation](https://inclusionai.github.io/AReaL/).
