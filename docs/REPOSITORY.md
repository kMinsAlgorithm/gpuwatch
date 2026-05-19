# Repository Guide

This repository contains the `gpuwatch` Python package, CLI, Textual UI, tests, screenshots,
and local workflow skill documentation.

## Main Commands

```bash
make install-dev
make test
make compile
make screenshots
make build
```

Equivalent commands without `make`:

```bash
python3 -m pip install -e ".[tui,dev]"
python3 -m unittest discover -s tests -v
python3 -m compileall -q gpuwatch tests tools
python3 tools/generate_screenshots.py
python3 -m build
```

## Important Paths

- `gpuwatch/`: installable Python package.
- `gpuwatch/cli.py`: CLI command registration and entry point.
- `gpuwatch/render/`: Rich and Textual UI renderers.
- `gpuwatch/backends/`: host, NVML, and fake sampling backends.
- `gpuwatch/training/`: heartbeat writer and log parser.
- `tests/`: unit tests.
- `docs/screenshots/`: generated responsive SVG screenshots.
- `tools/generate_screenshots.py`: deterministic screenshot generator.
- `skills/gpuwatch/`: Codex skill for GPU inspection workflows.

## Git Note

In this workspace, `.git` is currently an empty read-only directory, so `git status` does not
recognize the folder as a repository. If you want this directory to become a normal git repo,
remove that placeholder outside the read-only mount and run:

```bash
git init
git add .
git commit -m "Initial gpuwatch package"
```
