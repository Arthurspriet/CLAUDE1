"""HuggingFace GGUF-to-Ollama import pipeline.

Downloads GGUF models from HuggingFace, auto-selects quantization based on
available VRAM, and imports them into Ollama so they work as native models.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from claude1.config import MODELS_DIR, HF_TOKEN


# ── Quantization ranking (best quality first) ────────────────────────────────

QUANT_RANK: list[str] = [
    "F16", "BF16",
    "Q8_0",
    "Q6_K", "Q6_K_L",
    "Q5_K_M", "Q5_K_L", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_L", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_M", "Q3_K_L", "Q3_K_S",
    "Q2_K", "Q2_K_S",
    "IQ4_XS", "IQ4_NL",
    "IQ3_XXS", "IQ3_XS", "IQ3_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_M",
    "IQ1_S", "IQ1_M",
]

# Map quant tag to its quality rank (lower = better)
_QUANT_ORDER = {q.upper(): i for i, q in enumerate(QUANT_RANK)}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GGUFFile:
    """A single GGUF file in a HuggingFace repo."""
    filename: str
    size_bytes: int
    quant: str  # extracted quantization type (e.g. "Q4_K_M")

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def quality_rank(self) -> int:
        """Lower is better quality."""
        return _QUANT_ORDER.get(self.quant.upper(), 999)


@dataclass
class HFSearchResult:
    """A HuggingFace model repo from search results."""
    repo_id: str
    downloads: int
    likes: int
    tags: list[str]


@dataclass
class ImportProgress:
    """Progress update during import."""
    stage: str  # "search", "detect_vram", "select", "download", "import", "done", "error"
    message: str
    pct: float = 0.0  # 0-100 for download progress


# ── VRAM detection ────────────────────────────────────────────────────────────

def detect_vram() -> int:
    """Detect available GPU VRAM in MB. Returns 0 if no GPU found."""
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Sum all GPUs
            total = 0
            for line in result.stdout.strip().splitlines():
                try:
                    total += int(line.strip())
                except ValueError:
                    continue
            if total > 0:
                return total
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try rocm-smi for AMD GPUs
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if "total" in line.lower():
                    # Try to extract MB value
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        val = int(numbers[-1])
                        # If value seems to be in bytes, convert
                        if val > 1_000_000:
                            return val // (1024 * 1024)
                        return val
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0


def detect_ram() -> int:
    """Detect total system RAM in MB."""
    try:
        import resource
        # Linux: read /proc/meminfo
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Value is in kB
                    kb = int(line.split()[1])
                    return kb // 1024
    except (OSError, ValueError):
        pass

    # Fallback: try psutil
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        pass

    # Last resort: assume 16GB
    return 16384


def get_memory_budget() -> tuple[int, str]:
    """Get memory budget in MB and a description of what was detected.

    Returns (budget_mb, description).
    For GPU: uses VRAM with 20% headroom for KV cache.
    For CPU-only: uses RAM / 2.
    """
    vram = detect_vram()
    if vram > 0:
        budget = int(vram * 0.80)  # 20% headroom for KV cache + overhead
        return budget, f"GPU VRAM: {vram} MB (budget: {budget} MB after headroom)"

    ram = detect_ram()
    budget = ram // 2
    return budget, f"No GPU detected. RAM: {ram} MB (budget: {budget} MB = RAM/2)"


# ── HuggingFace API helpers ──────────────────────────────────────────────────

def _get_hf_api():
    """Lazily import and return HfApi instance."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub")
    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return HfApi(**kwargs)


def search_hf_gguf(query: str, limit: int = 15) -> list[HFSearchResult]:
    """Search HuggingFace for repos containing GGUF models.

    Args:
        query: Search query (e.g. "dolphin uncensored")
        limit: Max results to return

    Returns:
        List of HFSearchResult sorted by downloads
    """
    api = _get_hf_api()

    # Search for models; add "gguf" to query to bias results
    search_query = f"{query} gguf" if "gguf" not in query.lower() else query

    results = []
    try:
        models = api.list_models(
            search=search_query,
            sort="downloads",
            direction=-1,
            limit=limit * 3,  # fetch extra since we filter
        )

        for model in models:
            model_id = model.id if hasattr(model, 'id') else str(model)
            tags = list(model.tags) if hasattr(model, 'tags') and model.tags else []
            downloads = model.downloads if hasattr(model, 'downloads') else 0
            likes = model.likes if hasattr(model, 'likes') else 0

            # Filter: must have "gguf" in tags or model name
            is_gguf = (
                "gguf" in [t.lower() for t in tags]
                or "gguf" in model_id.lower()
            )
            if not is_gguf:
                continue

            results.append(HFSearchResult(
                repo_id=model_id,
                downloads=downloads,
                likes=likes,
                tags=tags,
            ))

            if len(results) >= limit:
                break

    except Exception as e:
        raise RuntimeError(f"HuggingFace search failed: {e}")

    return results


def list_gguf_files(repo_id: str) -> list[GGUFFile]:
    """List all GGUF files in a HuggingFace repo with their sizes and quant types.

    Args:
        repo_id: HuggingFace repo ID (e.g. "bartowski/Dolphin3.0-Llama3.1-8B-GGUF")

    Returns:
        List of GGUFFile sorted by quality (best first)
    """
    api = _get_hf_api()

    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        raise RuntimeError(f"Cannot list files in {repo_id}: {e}")

    # Get file sizes via repo info
    try:
        repo_info = api.repo_info(repo_id, files_metadata=True)
        file_sizes = {}
        if repo_info.siblings:
            for sibling in repo_info.siblings:
                if sibling.rfilename and sibling.size is not None:
                    file_sizes[sibling.rfilename] = sibling.size
    except Exception:
        file_sizes = {}

    gguf_files = []
    for fname in files:
        if not fname.lower().endswith(".gguf"):
            continue

        # Extract quantization from filename
        quant = _extract_quant(fname)
        size_bytes = file_sizes.get(fname, 0)

        gguf_files.append(GGUFFile(
            filename=fname,
            size_bytes=size_bytes,
            quant=quant,
        ))

    # Sort by quality rank (best quality first)
    gguf_files.sort(key=lambda f: f.quality_rank)
    return gguf_files


def _extract_quant(filename: str) -> str:
    """Extract quantization type from a GGUF filename.

    Examples:
        'model-Q4_K_M.gguf' -> 'Q4_K_M'
        'model.Q5_K_S.gguf' -> 'Q5_K_S'
        'model-f16.gguf' -> 'F16'
    """
    # Remove .gguf extension
    base = filename.rsplit(".gguf", 1)[0]

    # Try matching known quant patterns (case-insensitive)
    for quant in QUANT_RANK:
        pattern = re.compile(re.escape(quant), re.IGNORECASE)
        if pattern.search(base):
            return quant

    # Fallback: try common patterns
    match = re.search(r'[._-]((?:I?Q|F|BF)\d[A-Z0-9_]*)', base, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "unknown"


# ── Quantization selection ───────────────────────────────────────────────────

def pick_best_quant(files: list[GGUFFile], budget_mb: int) -> GGUFFile | None:
    """Pick the best quality GGUF file that fits in the memory budget.

    Uses file_size * 1.2 as the estimated runtime memory requirement
    (accounts for KV cache overhead during inference).

    Args:
        files: List of GGUFFile sorted by quality
        budget_mb: Available memory budget in MB

    Returns:
        Best GGUFFile that fits, or None if nothing fits
    """
    budget_bytes = budget_mb * 1024 * 1024

    # Files are already sorted by quality (best first)
    for f in files:
        estimated_runtime = f.size_bytes * 1.2
        if estimated_runtime <= budget_bytes:
            return f

    # Nothing fits at 1.2x, try with just the raw file size
    for f in files:
        if f.size_bytes <= budget_bytes:
            return f

    # Return smallest file as last resort (user can decide)
    if files:
        return min(files, key=lambda f: f.size_bytes)

    return None


# ── Download ─────────────────────────────────────────────────────────────────

def download_gguf(
    repo_id: str,
    filename: str,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """Download a GGUF file from HuggingFace to the local models directory.

    Args:
        repo_id: HuggingFace repo ID
        filename: GGUF filename within the repo
        progress_callback: Optional callback with download progress (0.0-100.0)

    Returns:
        Path to the downloaded file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(MODELS_DIR),
    }
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    try:
        path = hf_hub_download(**kwargs)
        return Path(path)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")


# ── Modelfile generation ─────────────────────────────────────────────────────

def create_modelfile(gguf_path: Path, num_ctx: int = 8192) -> Path:
    """Generate an Ollama Modelfile for the GGUF model.

    Args:
        gguf_path: Path to the downloaded GGUF file
        num_ctx: Context window size

    Returns:
        Path to the generated Modelfile
    """
    modelfile_content = f"""FROM {gguf_path.resolve()}

PARAMETER num_ctx {num_ctx}
PARAMETER temperature 0.7

TEMPLATE \"\"\"{{{{- if .System }}}}
{{{{ .System }}}}
{{{{- end }}}}
{{{{- range .Messages }}}}
{{{{- if eq .Role "user" }}}}
<|user|>
{{{{ .Content }}}}
{{{{- else if eq .Role "assistant" }}}}
<|assistant|>
{{{{ .Content }}}}
{{{{- end }}}}
{{{{- end }}}}
<|assistant|>
\"\"\"
"""

    modelfile_path = gguf_path.parent / f"{gguf_path.stem}.Modelfile"
    modelfile_path.write_text(modelfile_content)
    return modelfile_path


# ── Ollama import ────────────────────────────────────────────────────────────

def derive_model_name(repo_id: str, quant: str) -> str:
    """Derive a clean Ollama model name from repo_id and quant.

    Examples:
        'bartowski/Dolphin3.0-Llama3.1-8B-GGUF', 'Q4_K_M'
            -> 'dolphin3.0-llama3.1-8b:Q4_K_M'
    """
    # Get the model part (after org/)
    if "/" in repo_id:
        name = repo_id.split("/", 1)[1]
    else:
        name = repo_id

    # Strip common suffixes
    for suffix in ["-GGUF", "-gguf", "_GGUF", "_gguf"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Lowercase and clean
    name = name.lower().replace(" ", "-")

    # Remove any chars that Ollama doesn't like
    name = re.sub(r'[^a-z0-9._-]', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')

    return f"{name}:{quant}"


def import_to_ollama(model_name: str, modelfile_path: Path) -> str:
    """Import a model into Ollama using `ollama create`.

    Args:
        model_name: Name for the Ollama model
        modelfile_path: Path to the Modelfile

    Returns:
        Output from ollama create
    """
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"ollama create failed: {error}")
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("ollama command not found. Is Ollama installed?")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ollama create timed out (>5 min). The model file may be too large.")


# ── Full pipeline ────────────────────────────────────────────────────────────

def full_import(
    repo_id: str,
    model_name: str | None = None,
    num_ctx: int = 8192,
    progress_callback: Callable[[ImportProgress], None] | None = None,
) -> str:
    """End-to-end import: list files, pick quant, download, import into Ollama.

    Args:
        repo_id: HuggingFace repo ID (e.g. "bartowski/Dolphin3.0-Llama3.1-8B-GGUF")
        model_name: Optional custom Ollama model name
        num_ctx: Context window size
        progress_callback: Optional callback for progress updates

    Returns:
        The Ollama model name that was created
    """
    def _progress(stage: str, message: str, pct: float = 0.0):
        if progress_callback:
            progress_callback(ImportProgress(stage=stage, message=message, pct=pct))

    # Step 1: List GGUF files in repo
    _progress("search", f"Scanning {repo_id} for GGUF files...")
    try:
        files = list_gguf_files(repo_id)
    except Exception as e:
        _progress("error", str(e))
        raise

    if not files:
        msg = f"No GGUF files found in {repo_id}"
        _progress("error", msg)
        raise RuntimeError(msg)

    _progress("search", f"Found {len(files)} GGUF file(s)")

    # Step 2: Detect VRAM / memory budget
    _progress("detect_vram", "Detecting available memory...")
    budget_mb, mem_desc = get_memory_budget()
    _progress("detect_vram", mem_desc)

    # Step 3: Pick best quantization
    _progress("select", "Selecting optimal quantization...")
    selected = pick_best_quant(files, budget_mb)
    if selected is None:
        msg = "No GGUF file fits available memory. Consider freeing up GPU memory or using a smaller model."
        _progress("error", msg)
        raise RuntimeError(msg)

    fits_budget = (selected.size_bytes * 1.2) <= (budget_mb * 1024 * 1024)
    fit_note = "" if fits_budget else " (WARNING: may not fit in memory)"
    _progress("select", f"Selected: {selected.quant} ({selected.size_gb:.1f} GB){fit_note}")

    # Step 4: Derive model name
    if model_name is None:
        model_name = derive_model_name(repo_id, selected.quant)

    # Step 5: Download
    _progress("download", f"Downloading {selected.filename} ({selected.size_gb:.1f} GB)...")
    try:
        gguf_path = download_gguf(repo_id, selected.filename)
    except Exception as e:
        _progress("error", f"Download failed: {e}")
        raise
    _progress("download", f"Downloaded to {gguf_path}", pct=100.0)

    # Step 6: Create Modelfile
    _progress("import", "Generating Modelfile...")
    modelfile_path = create_modelfile(gguf_path, num_ctx=num_ctx)

    # Step 7: Import into Ollama
    _progress("import", f"Running: ollama create {model_name}...")
    try:
        output = import_to_ollama(model_name, modelfile_path)
    except Exception as e:
        _progress("error", f"Ollama import failed: {e}")
        raise

    _progress("done", f"Model '{model_name}' is ready! Use with: /model {model_name}")

    # Clean up Modelfile (keep the GGUF)
    try:
        modelfile_path.unlink()
    except OSError:
        pass

    return model_name

