## LLM Experiments

A collection of small, focused experiments exploring large language models and related systems. The aim is to study how these models behave internally, how they respond to different prompting strategies, and how architectural or design choices affect their outputs.

The repository is intentionally exploratory. Each directory contains a self-contained experiment, ranging from representation analysis and prompting behavior to system-level ideas such as routing, scaling, and tooling. Over time, this will grow to include a wider range of investigations across machine learning, LLM systems, and adjacent areas.

The goal is not to build production-ready systems, but to develop a deeper, more practical understanding of how these models work in practice.

## Setup

1. Clone the repo 

    ```bash
        git clone https://github.com/Vishvam10/llm-experiments
    ```

2. Create virtual environment using [uv](https://docs.astral.sh/uv/getting-started/installation/) :

    ```bash
        uv venv --python 3.12
    ```

3. Install dependecies, run following command from root directory.

    ```bash
        uv sync

        # To upgrade all packages to their latest compatible versions
        uv sync --upgrade
    ```

## Development

1. Linting

    ```bash
        # show fixes (do this)
        ruff check --config pyproject.toml

        # should be enough (do this).
        ruff check --fix --config pyproject.toml

        ruff check --unsafe-fixes --config pyproject.toml        # only if necessary
        ruff check --fix --unsafe-fixes --config pyproject.toml  # only if necessary
    ```

> [!NOTE]  
> Manually correct your code and abide to whatever `ruff check` suggests.
> Running `ruff check --fix` can only do so much.

2. Formatting

   ```bash
      ruff check --select I --fix . --config pyproject.toml
      ruff format --config pyproject.toml
   ```

> [!NOTE]
> We do the `check` step as well to enforce `isort` (import sort) as well
> while formatting
