# Contributing

## Development setup

```bash
python -m pip install -e ".[dev]"
```

## Required checks

```bash
python -m pytest
python -m compileall -q heraclitus
ruff check .
mypy heraclitus
python -m build
python -m twine check dist/*
```

Every mathematical change must include a test that exercises the declared invariant or equation. Every state transition change must preserve causality, batch isolation, mask semantics, and chunk equivalence.

Public APIs require type annotations and docstrings. Repository text and source comments use ASCII punctuation.
