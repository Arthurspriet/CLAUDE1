#!/usr/bin/env python3
"""Claude1 - Local Coding Assistant. Entry point."""

import sys
import os

import click

from config import AppConfig, DEFAULT_MODEL, HF_TOKEN, HF_ENDPOINT, parse_model_spec
from model_profiles import get_profile


@click.command()
@click.option("-m", "--model", default=DEFAULT_MODEL, help="Model to use (prefix with 'hf:' for HuggingFace, e.g. hf:meta-llama/Meta-Llama-3-8B-Instruct)")
@click.option("-y", "--auto-accept", is_flag=True, help="Auto-accept all tool confirmations")
@click.option("-d", "--dir", "working_dir", default=None, help="Working directory")
@click.option("-p", "--print", "print_mode", is_flag=True, help="Non-interactive print mode (output raw text, exit)")
@click.option("-v", "--verbose", is_flag=True, help="Show debug info (API calls, tool timing)")
@click.option("-r", "--resume", is_flag=True, help="Resume last auto-saved session")
@click.option("--temp", "temperature", type=float, default=None, help="Set temperature (0.0-2.0)")
@click.option("--agents", is_flag=True, help="Enable multi-agent mode")
@click.argument("prompt", required=False, default=None)
def main(model: str, auto_accept: bool, working_dir: str | None, print_mode: bool,
         verbose: bool, resume: bool, temperature: float | None, agents: bool, prompt: str | None):
    """Claude1 - A local coding assistant powered by Ollama and HuggingFace."""
    # Resolve working directory
    if working_dir:
        wd = os.path.abspath(working_dir)
        if not os.path.isdir(wd):
            click.echo(f"Error: Directory not found: {wd}", err=True)
            sys.exit(1)
    else:
        wd = os.getcwd()

    # Determine provider from model spec
    provider, model_id = parse_model_spec(model)

    model_info = None
    hf_endpoint = ""

    if provider == "huggingface":
        # Validate huggingface_hub is installed
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            click.echo("Error: huggingface_hub is not installed. Run: pip install huggingface_hub", err=True)
            sys.exit(1)

        # Warn if no HF_TOKEN set (cloud API needs it)
        endpoint = HF_ENDPOINT
        if not HF_TOKEN and not endpoint:
            click.echo("Warning: HF_TOKEN not set. Cloud API calls may fail. Set HF_TOKEN env var or HF_ENDPOINT for local TGI.", err=True)

        hf_endpoint = endpoint

        # Optionally health-check a local TGI endpoint
        if endpoint:
            try:
                import urllib.request
                req = urllib.request.Request(f"{endpoint.rstrip('/')}/health", method="GET")
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                click.echo(f"Warning: Cannot reach TGI endpoint at {endpoint}. Continuing anyway.", err=True)

    else:
        # Ollama provider — verify Ollama is running
        import ollama
        try:
            ollama.list()
        except Exception as e:
            click.echo(f"Error: Cannot connect to Ollama. Is it running?\n  {e}", err=True)
            click.echo("Start Ollama with: ollama serve", err=True)
            sys.exit(1)

        # Verify model exists
        try:
            models = ollama.list()
            available = [m.model for m in models.models]
            if model not in available:
                # Try partial match
                matches = [m for m in available if m.startswith(model.split(":")[0])]
                if matches:
                    click.echo(f"Model '{model}' not found. Did you mean one of: {', '.join(matches)}?", err=True)
                else:
                    click.echo(f"Model '{model}' not found. Available models: {', '.join(available)}", err=True)
                sys.exit(1)
        except Exception:
            pass  # Don't fail on model check - let it fail at chat time

        # Get model info for welcome banner
        try:
            info = ollama.show(model)
            details = info.get("details", {}) if isinstance(info, dict) else getattr(info, "details", None)
            if details:
                if isinstance(details, dict):
                    model_info = {
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization_level": details.get("quantization_level", ""),
                        "family": details.get("family", ""),
                    }
                else:
                    model_info = {
                        "parameter_size": getattr(details, "parameter_size", ""),
                        "quantization_level": getattr(details, "quantization_level", ""),
                        "family": getattr(details, "family", ""),
                    }
        except Exception:
            pass

    # Resolve model profile
    ollama_family = model_info.get("family", "") if model_info else None
    profile = get_profile(model, ollama_family=ollama_family)

    # Handle pipe/stdin input
    stdin_content = None
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read()
        # Reopen stdin from /dev/tty so prompt_toolkit works
        try:
            sys.stdin = open("/dev/tty", "r")
        except OSError:
            pass  # On systems without /dev/tty, print mode is expected

    # Combine stdin and prompt
    combined_prompt = None
    if stdin_content and prompt:
        combined_prompt = f"<stdin>\n{stdin_content}\n</stdin>\n\n{prompt}"
    elif stdin_content:
        combined_prompt = f"<stdin>\n{stdin_content}\n</stdin>\n\nPlease analyze the above input."
    elif prompt:
        combined_prompt = prompt

    # Validate print mode
    if print_mode and not combined_prompt:
        click.echo("Error: --print mode requires a prompt (positional argument or piped stdin)", err=True)
        sys.exit(1)

    # Create config and start REPL
    config = AppConfig(
        model=model,
        working_dir=wd,
        auto_accept_tools=auto_accept or print_mode,
        num_ctx=profile.num_ctx,
        temperature=temperature,
        verbose=verbose,
        agents_mode=agents,
        model_info=model_info,
        profile=profile,
        provider=provider,
        hf_endpoint=hf_endpoint,
    )

    # Add project dir to path for imports
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from repl import REPL
    repl = REPL(config)

    if print_mode:
        repl.run_print(combined_prompt)
    else:
        # Load autosave if --resume
        if resume:
            from session import get_latest_session
            messages = get_latest_session()
            if messages:
                repl.llm.messages = messages
                click.echo(f"Resumed session ({len(messages)} messages)", err=True)

        repl.run(initial_prompt=combined_prompt)


if __name__ == "__main__":
    # Ensure project directory is in sys.path for imports
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    main()
