# Contributing to FinanceBench-LLM

Thanks for your interest in contributing!

## Setup

```bash
git clone https://github.com/amitlals/FinanceBench-LLM.git
cd FinanceBench-LLM
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env with your API keys
```

## Code Style

We use **ruff** for linting and formatting:

```bash
ruff check src/ hf_space/       # Lint
ruff format src/ hf_space/      # Format
mypy src/                       # Type check
```

## Running Tests

```bash
pytest tests/ -v
```

All tests run offline (no API calls required).

## Notebook Conventions

- Notebooks are numbered sequentially (`1_` through `5_`)
- Each notebook depends on results from previous notebooks
- Use `from src.utils import ...` for shared utilities (backward-compatible)
- Clear all outputs before committing (keeps diffs clean)

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run `ruff check` and `pytest` locally
4. Open a PR using the template
5. Ensure CI passes

## Architecture Decisions

- **`src/` modules**: Each file has a single responsibility (see `src/__init__.py` for the full module map)
- **`config.yaml`**: All hyperparameters and constants are centralized here
- **Backward compatibility**: `src/utils.py` re-exports everything so existing notebook imports continue to work
