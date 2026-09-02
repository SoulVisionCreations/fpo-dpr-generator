#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANDOC="$BASE/venv/lib/python3.13/site-packages/pypandoc/files/pandoc"
TECTONIC="$BASE/venv/bin/tectonic"
STYLE="$BASE/pdf/dpr_pdf.yaml"
OUT="$BASE/output/pdf"

mkdir -p "$OUT"
# Tectonic writes its TeX bundle/cache under HOME. Keep that cache inside the
# project rather than the read-only user home of the execution environment.
export HOME="$BASE/.tectonic-home"
export XDG_CACHE_HOME="$HOME/.cache"
mkdir -p "$XDG_CACHE_HOME"

if [[ ! -x "$PANDOC" || ! -x "$TECTONIC" ]]; then
  echo "Pandoc or Tectonic is missing from the project venv." >&2
  exit 1
fi

declare -A SOURCES=(
  [rice_mill]="DPR_rice__gemma-4-12B-it_20260717_140144.md"
  [mustard_oil]="DPR_mustard_oil__gemma-4-12B-it_20260715_181805.md"
  [fruit_vegetable_grading]="DPR_graded_fruits_and_vegetables__gemma-4-12B-it_20260717_140243.md"
  [vermicompost]="DPR_vermicompost__gemma-4-12B-it_20260716_121131.md"
  [grain_storage]="DPR_grain_storage_services__gemma-4-12B-it_20260719_192817.md"
  [spice_processing]="DPR_spices__gemma-4-12B-it_20260720_115119.md"
  [custom_hiring_centre]="DPR_custom_hiring_services__gemma-4-12B-it_20260719_192700.md"
  [nursery_hardening]="DPR_hardened_saplings__gemma-4-12B-it_20260720_114912.md"
)

for key in "${!SOURCES[@]}"; do
  source="$BASE/output/${SOURCES[$key]}"
  target="$OUT/DPR_${key}.pdf"
  echo "[pdf] $(basename "$source") -> $(basename "$target")"
  "$PANDOC" "$source" --defaults="$STYLE" --pdf-engine="$TECTONIC" -o "$target"
done

echo "[done] PDFs written to $OUT"
