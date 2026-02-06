---
name: commit
description: "Stage changes and create a well-structured git commit"
allowed-tools: bash, read_file, grep_search, glob_search
---

## Instructions

Help the user create a well-structured git commit. Follow these steps:

1. Run `git status` to see what files have changed
2. Run `git diff` to review the actual changes (both staged and unstaged)
3. Analyze the changes and determine:
   - What type of change this is (feat, fix, refactor, docs, test, chore)
   - A clear, concise summary of what changed and why
4. Stage the appropriate files with `git add`
5. Create the commit with a well-formatted message following conventional commits:
   - First line: type(scope): short description (max 72 chars)
   - Blank line
   - Body: explain what and why (not how)

If $ARGUMENTS is provided, use it as context for the commit scope/message.

Do NOT push to remote unless explicitly asked.
