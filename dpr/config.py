"""Central configuration & paths for the FPO DPR pipeline."""
import os
import shutil
import sysconfig
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent           # <repo>/dpr
REPO_ROOT   = PACKAGE_DIR.parent                         # <repo>

# ---- data ----
DATA_DIR      = REPO_ROOT / "data"
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
MODEL_ID        = os.environ.get("DPR_MODEL_ID", "google/gemma-4-12B-it")
DEFAULT_ADAPTER = MODELS_DIR / "google_gemma-4-12B-it_v4" / "adapter"
MAX_NEW_TOKENS  = 4096
TEMPERATURE     = 0.7
TOP_P           = 0.9
DO_SAMPLE       = True
