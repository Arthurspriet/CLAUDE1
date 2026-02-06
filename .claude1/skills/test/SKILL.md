---
name: test
description: "Run project tests and analyze failures"
allowed-tools: bash, read_file, grep_search, glob_search
---

## Instructions

Detect the project's test framework, run the tests, and analyze the results. Follow these steps:

1. Detect the test framework by checking for:
   - Python: `pytest.ini`, `pyproject.toml` (pytest section), `setup.cfg`, test files
   - Node.js: `jest.config.js`, `vitest.config.ts`, `package.json` scripts
   - Go: `*_test.go` files
   - Rust: `cargo test`
   - Other: `Makefile` test targets
2. Run the appropriate test command
3. Analyze the output:
   - If all pass: summarize test count and coverage if available
   - If failures: for each failing test, explain what failed and suggest a fix
   - If errors: diagnose setup/configuration issues

If $ARGUMENTS is provided, use it to narrow the test scope (e.g., specific test file, test name pattern, or module).

Always show the full test command you ran so the user can reproduce it.
