#!/usr/bin/env bash
# One-shot status snapshot for the v3 gpt-5 finetune pipeline.
#   ./pipeline_status.sh
cd "$(dirname "$0")/../.." || exit 1   # repo root
FT=data/finetune
TARGET=40

echo "=== running processes ==="
pgrep -af "gpt_gen_payloads|teacher|sft_train" || echo "  (nothing running)"

echo
echo "=== STEP 1/1b: payload instances (target ${TARGET}/commodity) ==="
tot=$(ls $FT/instances/*.json 2>/dev/null | wc -l)
echo "  total instances: $tot"
ls $FT/instances/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null \
  | sed 's/__g[0-9]*\.json//' | sort | uniq -c

echo
echo "=== STEP 2: DPRs authored ==="
auth=$(ls $FT/teacher_out/*.md 2>/dev/null | wc -l)
echo "  authored: $auth / $tot instances"

echo
echo "=== STEP 3: v3 dataset (train/val.jsonl) ==="
wc -l $FT/train.jsonl $FT/val.jsonl 2>/dev/null || echo "  not built yet"

echo
echo "=== STEP 4: finetune adapter ==="
if ls -d models/*/adapter >/dev/null 2>&1; then
  ls -d models/*/adapter | sed "s/^/  present: /"
else
  echo "  not trained yet"
fi
squeue -u "$USER" 2>/dev/null | grep -v JOBID || true

echo
echo "=== log tails ==="
for l in payloads_v3_topup author_v3; do
  [ -f "$FT/logs/$l.log" ] && { echo "--- $l.log ---"; tail -n 3 "$FT/logs/$l.log"; }
done
