"""Central configuration & paths for the FPO DPR pipeline."""
import os
import shutil
import sysconfig
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent           # <repo>/dpr
REPO_ROOT   = PACKAGE_DIR.parent                         # <repo>


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


def _env_int(var: str, default: int) -> int:
    v = os.environ.get(var)
    return int(v) if v else default


# ---- data ----
# DPR_DATA_DIR matters in a container: REPO_ROOT is derived from this file's
# location, so a non-editable install would put `dpr/` in site-packages and leave
# `data/templates/*.json` unreachable.
DATA_DIR      = _env_path("DPR_DATA_DIR", REPO_ROOT / "data")
TEMPLATES_DIR = DATA_DIR / "templates"
PAYLOADS_DIR  = DATA_DIR / "payloads"
GENERATED_DIR = DATA_DIR / "generated"
FINETUNE_DIR  = DATA_DIR / "finetune"
EXAMPLES_DIR  = DATA_DIR / "examples"
GHEE_FIXTURE  = DATA_DIR / "ghee_instance.json"

# ---- outputs / assets / models ----
OUTPUT_DIR = REPO_ROOT / "output"
PDF_ASSETS = REPO_ROOT / "assets" / "pdf"
MODELS_DIR = REPO_ROOT / "models"

# ---- PDF toolchain ----
def _tool(name: str, *bundled: Path) -> Path:
    """Resolve an external binary: a bundled copy if present, else PATH.

    Falls back to the first bundled candidate so a failed lookup still yields a
    concrete path for the error message rather than blowing up at import time.
    """
    for cand in bundled:
        if cand.is_file():
            return cand
    found = shutil.which(name)
    return Path(found) if found else bundled[0]


# pypandoc-binary ships pandoc inside site-packages; plain pypandoc does not.
_PURELIB  = Path(sysconfig.get_paths()["purelib"])
PANDOC    = _tool("pandoc", _PURELIB / "pypandoc" / "files" / "pandoc")
TECTONIC  = _tool("tectonic", REPO_ROOT / "venv" / "bin" / "tectonic")
PDF_STYLE = PDF_ASSETS / "dpr_pdf.yaml"
PDF_NOTOC = PDF_ASSETS / "_notoc.yaml"

# ---- model (narrative lane) ----
# Single source of truth for both the CLI and the HTTP service. MAX_NEW_TOKENS is
# 3500 (not 4096) because that is what the shipped v4 adapter was evaluated at.
MODEL_ID        = os.environ.get("DPR_MODEL_ID", "google/gemma-4-12B-it")
MODEL_REVISION  = os.environ.get("DPR_MODEL_REVISION")     # pin to make runs reproducible
DEFAULT_ADAPTER = _env_path("DPR_ADAPTER_DIR", MODELS_DIR / "google_gemma-4-12B-it_v4" / "adapter")
MAX_NEW_TOKENS  = _env_int("DPR_MAX_NEW_TOKENS", 3500)
RETRIES         = _env_int("DPR_RETRIES", 2)
TEMPERATURE     = 0.7            # retries only; attempt 0 is greedy
TOP_P           = 0.9
DO_SAMPLE       = True

# The model ends its turn with <turn|> (106), NOT <eos> (1). generation_config.json
# lists eos_token_id = [1, 106, 50]; the HF tokenizer alone reports only 1. Any
# backend that does not honour all three will generate to MAX_NEW_TOKENS every call.
STOP_TOKEN_IDS  = [1, 106, 50]
# <audio|>, <image|> — suppressed for text-only inference (HF does this via
# generation_config's suppress_tokens; other backends must be told explicitly).
SUPPRESS_TOKEN_IDS = [258883, 258882]
