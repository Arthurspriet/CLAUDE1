---
name: review
description: "Review code changes for bugs, style, and improvements"
allowed-tools: bash, read_file, grep_search, glob_search
---

## Instructions

Review the current code changes for potential issues. Follow these steps:

1. Run `git diff` to see unstaged changes, and `git diff --staged` for staged changes
2. For each changed file, analyze:
   - **Bugs**: Logic errors, off-by-one, null/None handling, race conditions
   - **Security**: Injection, XSS, hardcoded secrets, unsafe deserialization
   - **Style**: Naming conventions, code organization, unnecessary complexity
   - **Performance**: N+1 queries, unnecessary allocations, missing indexes
3. Provide feedback organized by severity:
   - Critical: Must fix before committing
   - Warning: Should consider fixing
   - Suggestion: Nice-to-have improvements

If $ARGUMENTS is provided, focus the review on those specific aspects or files.

Be constructive and specific. Reference line numbers and suggest fixes.
