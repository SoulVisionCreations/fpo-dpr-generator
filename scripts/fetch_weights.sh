#!/usr/bin/env bash
# Download the merged DPR weights from GHCR, resumably.
#
# `oras pull` has no resume: one dropped connection and all 22 GB start again.
# On a flaky link that can mean never finishing. GHCR serves blobs with range
# support (verified: HTTP 206), so curl -C - picks up where it stopped.
#
# Safe to re-run: completed files are verified once, stamped, and skipped after.
#
#   ./scripts/fetch_weights.sh
#   DPR_WEIGHTS=/data/weights ./scripts/fetch_weights.sh
#   until ./scripts/fetch_weights.sh; do sleep 30; done      # ride out an outage
set -euo pipefail

REF="${1:-ghcr.io/soulvisioncreations/dpr-weights:v4}"
DEST="${DPR_WEIGHTS:-$HOME/dpr-weights}"

REG="${REF%%/*}"                 # ghcr.io
rest="${REF#*/}"                 # soulvisioncreations/dpr-weights:v4
REPO="${rest%:*}"                # soulvisioncreations/dpr-weights
TAG="${rest##*:}"                # v4

mkdir -p "$DEST"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# --http1.1 throughout: the HTTP/2 multiplexing is what produces the
# "stream ID N; PROTOCOL_ERROR" failures against GHCR on an unstable link.
CURL=(curl -fsSL --http1.1 --retry 10 --retry-delay 5 --retry-all-errors
      --connect-timeout 20)

# sha256sum on Linux, shasum on macOS. The target is a Linux box, but keeping
# this portable is what lets the script be exercised before it gets there.
if command -v sha256sum >/dev/null; then
  sha256of() { sha256sum "$1" | cut -d' ' -f1; }
else
  sha256of() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

# Anonymous, for a public package. Tokens are short-lived, so take a fresh one
# per blob rather than one for the whole 22 GB.
token() {
  "${CURL[@]}" "https://${REG}/token?scope=repository:${REPO}:pull&service=${REG}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
}

echo "source : ${REF}"
echo "dest   : ${DEST}"

"${CURL[@]}" -H "Authorization: Bearer $(token)" \
  -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  "https://${REG}/v2/${REPO}/manifests/${TAG}" -o "$TMP/manifest.json"

python3 - "$TMP/manifest.json" > "$TMP/layers.tsv" <<'PYEOF'
import json, sys
for l in json.load(open(sys.argv[1]))["layers"]:
    title = l.get("annotations", {}).get("org.opencontainers.image.title")
    if title:                      # unnamed layers are not files we want on disk
        print(f"{l['digest']}\t{l['size']}\t{title}")
PYEOF

total=$(awk -F'\t' '{s+=$2} END {printf "%.2f", s/1073741824}' "$TMP/layers.tsv")
echo "files  : $(wc -l < "$TMP/layers.tsv") (${total} GiB)"
echo

while IFS=$'\t' read -r digest size title; do
  out="$DEST/$title"
  stamp="$DEST/.${title}.ok"

  if [[ -f "$stamp" ]]; then
    echo "skip   $title (verified earlier)"
    continue
  fi

  have=0; [[ -f "$out" ]] && have=$(stat -c %s "$out" 2>/dev/null || stat -f %z "$out")
  if (( have < size )); then
    (( have > 0 )) && echo "resume $title ($(( have * 100 / size ))% done)" \
                   || echo "get    $title"
    # -C - continues from the local file's length. A failure here leaves the
    # partial file in place, so the next run resumes rather than restarts.
    "${CURL[@]}" -C - -H "Authorization: Bearer $(token)" \
      -o "$out" "https://${REG}/v2/${REPO}/blobs/${digest}"
  fi

  # Truncation is caught by the size check; corruption only by the digest.
  echo -n "verify $title ... "
  got="sha256:$(sha256of "$out")"
  if [[ "$got" != "$digest" ]]; then
    echo "MISMATCH"
    echo "  expected $digest" >&2
    echo "  got      $got" >&2
    echo "  removing it; re-run this script to fetch it again" >&2
    rm -f "$out"
    exit 1
  fi
  echo "ok"
  : > "$stamp"
done < "$TMP/layers.tsv"

echo
echo "complete: $(du -sh "$DEST" | cut -f1) in $DEST"
