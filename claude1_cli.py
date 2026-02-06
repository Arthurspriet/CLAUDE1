#!/usr/bin/env python3
"""CLI entry point for claude1 command."""
import sys
import os

# Ensure project directory is in sys.path for local imports
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from main import main

def cli():
    main()

if __name__ == "__main__":
    cli()
