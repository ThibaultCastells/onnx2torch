# onnx2torch

Convert ONNX models to torch.export ExportedProgram bundles (.pt2, ATen dialect).
It is assumed that the input ONNX model's shapes are already known.

# core instructions

- Environment setup:
    Run source `.venv/bin/activate` if the environment isn't activated yet.
- Scripts & configs:
    - Prefer configuration-driven scripts (YAML).
    - Use the `--cfg` CLI argument to specify the config file path.
    - By default, the configs are in `cfg/`
- Change logs
    - run `python logs/contrib_logs/log_new_contrib.py` to create a new log entry, then fill in the Summary and Files lines before committing. Example of log content:
    ``` markdown
    ## 2025-09-26

    ### 2025-09-26 01:14 UTC
    - Summary: Removed legacy visit limit alias; refreshed CLI docs/tests and daily log structure
    - Files: src/hf_fetch_and_convert/search_models.py, unit_tests/test_list_candidates.py, unit_tests/test_search_models.py, README.md, logs/2509.md
    ```
- Run logs:
    - All run logs should be saved in `logs/run_logs/`.
    - Create one folder per run, containing:
        - A run log file
        - The config file used for that run
        - Any other lightweight, useful data related to the run
- Codebase organization:
    - Keep the codebase clean, readable, and modular.
    - Avoid files that are too long; split functionality into smaller, focused modules.
    - Ensure functions and classes are concise and easy to understand.
    - Strive for maintainability and clarity over cleverness.
- Doc maintenance
    - When adding support for a new operator, update `operators.md` accordingly
- Testing:
    - Test all changes before committing.
    - Add fast-to-run unit tests for key tools in `unit_tests/`.
- Linting & formatting:
    - Run ruff when you are done (`format` and `check --fix`)
