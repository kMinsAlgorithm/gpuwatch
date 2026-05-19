.PHONY: install install-dev test compile screenshots build clean

install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[tui,dev]"

test:
	python3 -m unittest discover -s tests -v

compile:
	python3 -m compileall -q gpuwatch tests tools

screenshots:
	python3 tools/generate_screenshots.py

build:
	python3 -m build

clean:
	python3 -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]; [shutil.rmtree(p, ignore_errors=True) for p in (pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('.mypy_cache'))]"
