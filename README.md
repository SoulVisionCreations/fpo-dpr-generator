# FPO DPR Generator

A pipeline that turns a short web form into a complete, bankable **Detailed Project
Report (DPR)** for an Indian **Farmer Producer Organization (FPO)** — NABARD / NIFTEM
PM-FME format, across **12 agri-enterprise business lines**.

Its defining property (**two lanes**): a deterministic **engine computes every
financial number**, a fine-tuned **LLM writes only the prose**, and every figure the
model types is **verified against the engine** before the report ships. No
AI-invented number ever reaches a DPR.

```
form JSON ─┬─▶ engine  (ALL numbers) ───────────────▶ exact tables ─┐
           │                                                        ├─▶ assemble + verify ─▶ DPR ─▶ PDF
           └─▶ LLM     (narrative prose + {{TABLE}} placeholders) ──┘
```

## What a DPR contains

Every generated report follows the same NABARD / PM-FME structure — 6 sections, with
§5 carrying 19 costed sub-schedules:

| § | Section | Written by |
|---|---|---|
| 1 | Project Summary | engine |
| 2 | Product, Market & Value Chain | **model** (`###` sub-headings: Introduction · Uses & Benefits · Market Potential · Raw Material) |
| 3 | Basis & Presumptions | engine |
| 4 | Process Flow Chart | engine |
| 5 | Economics of the Project | engine (19 schedules, below) |
| 6 | Limitations of the Model DPR | fixed text |
| — | Disclaimer | fixed text |

**§5 — Economics of the Project**

```
5.1  Key Operating Parameters          5.11 Salaries & Wages
5.2  Capacity, Utilisation, Output     5.12 Power Requirement
5.3  Premises                          5.13 Depreciation
5.4  Machinery & Equipment             5.14 Repair & Maintenance
5.5  Misc Fixed Assets                 5.15 Profitability Analysis
5.6  Total Cost of Project             5.16 Break-Even Point
5.7  Means of Finance                  5.17 Projected Balance Sheet
5.8  Term Loan                         5.18 Cash-Flow Statement
5.9  Repayment & Interest Schedule     5.19 Debt-Service Coverage Ratio
5.10 Working Capital
```

All money figures are in **Rs. Lakh**, projected over **5 years**. The model authors the
headings and §2 prose and emits `{{TABLE:<id>}}` placeholders everywhere else; `assemble()`
swaps in the engine's exact tables and `verify()` rejects any number the model typed that
the engine didn't produce. Canonical list: `SECTION_ORDER` in `dpr/render/sections.py`.

## Repository layout

```
dpr/                     ← the core library (production; pure-Python except the model lane)
  config.py                paths + model settings (single source of truth)
  form_to_instance.py      form payload → engine instance  (convert_website / convert_nested)
  authoring.py             build_full_prompt() — deterministic prompt, NO LLM
  make_pdf.py              markdown → PDF (pandoc + tectonic)
  engine/financial_engine.py   all numbers        engine/scheme_norms.py   scheme rules
  render/sections.py           engine → tables    render/assemble.py        inject tables + verify()
scripts/run_pipeline.py   ← USAGE: one command, flat JSON → DPR PDF
training/                 ← how the model was built (teacher, dataset, SFT, eval)
  teacher.py  gpt_gen_payloads.py  gen_dataset.py  sft_train.py  eval_ft.py  score_dpr.py
  slurm/      launch_sft.sh  launch_eval.sh  author_v3.sh  pipeline_status.sh
data/         templates/ (12 lines) · payloads/ (seeds) · examples/ (sample form JSON) · finetune/
assets/pdf/   pandoc/tectonic styles
schema/       form payload contract (dpr_input.schema.json + example)
docs/         flow.md · FILE_FLOW.md · CONFIG_REFERENCE.md · FINANCIAL_ENGINE.md · schema/
```

Not in git (see `.gitignore`): `models/` (weights — see **Get the model** below), `output/`
and `logs/` (run artifacts), `venv/`, `.env`.

## Install
```bash
python3 -m venv venv && . venv/bin/activate    # macOS has no bare `python`
pip install -e .              # core engine/render/convert — pure stdlib, no GPU
pip install -e ".[pdf]"       # + markdown → PDF   (pandoc, bundled by pypandoc-binary)
pip install -e ".[model]"     # + narrative lane   (torch / transformers / peft, GPU)
```
`tectonic` is **not** bundled — install it for the PDF lane:
```bash
brew install tectonic          # macOS
apt install tectonic           # Debian/Ubuntu  (or: cargo install tectonic)
```
`dpr/config.py` finds both binaries via the venv first, then `PATH`, so either source works.
Tectonic caches LaTeX packages in `.tectonic-home/` on first run.

Requires **Python ≥ 3.10**. Put `OPENAI_API_KEY`or any frontier model's api key  in `.env` (teacher/dataset lane only —
see `.env.example`). *(Not installing? Everything also runs from the repo root, which is on
`sys.path` for `python -m …` and for `scripts/run_pipeline.py`.)  
(OpenAI api key is used here as an example since it was the frontier model used during training)*

## Get the model
The v4 LoRA adapter (~81 MB) is **not in this repo**. Download it and place it at
`models/google_gemma-4-12B-it_v4/adapter/`, or point `--adapter` at wherever you put it:
```bash
python scripts/run_pipeline.py --payload <form.json> --adapter /path/to/adapter
```
The base model `google/gemma-4-12B-it` (~24 GB, ungated) is pulled from Hugging Face on
first run. Inference needs a GPU with **≥ 24 GB** free — the adapter is LoRA-only
(r=16 over `q,k,v,o_proj`) and loads on top of the bf16 base.

## Usage — generate a DPR (flat form JSON → PDF)
```bash
python scripts/run_pipeline.py --payload data/examples/mango_juice.json
# → output/DPR_<product>_<timestamp>.pdf   (+ .md alongside)
```
**Want markdown only?** Add `--md-only` — it stops after the `.md` and skips the PDF step
entirely, so pandoc/tectonic aren't needed.

Other options: `--out <path.pdf>`, `--adapter none` (base model), `--adapter <dir>`,
`--retries N`, `--max-new-tokens N`.
It runs `convert → compute → model → assemble+verify (retry if unclean) → PDF`. Only the
model step needs a GPU.

## Serve — HTTP endpoint (markdown only)
A single FastAPI process with vLLM in-process; synchronous, markdown only (no PDF).
```bash
docker build -t dpr-api .
docker run -d --gpus all --ipc=host -p 8000:8000 \
  -e DPR_API_KEY=... \
  -v /mnt/dpr/hf:/models/hf \
  -v /mnt/dpr/adapter:/adapters/v4:ro dpr-api

curl -X POST localhost:8000/v1/dpr:generate -H "X-API-Key: $DPR_API_KEY" \
     -H 'content-type: application/json' -d '{"payload": {...}}'
```
`POST /v1/dpr:generate` returns the markdown plus the engine's verification report
and any silent-defaulting warnings. `/health` is liveness, `/ready` is readiness —
the engine takes minutes to load 24 GB. Full reference: [docs/serving.md](docs/serving.md).

## Retrain (v-next) — data → finetune → eval
```bash
# 1) generate viable, commodity-varied instances (frontier teacher; key auto-loaded from .env)
python -m training.gpt_gen_payloads --provider openai --model gpt-5 --target 60 --concurrency 8 --max-rounds 12
# 2) author a full DPR per instance with the teacher (resumable)
./training/slurm/author_v3.sh gpt-5 8
# 3) assemble+verify each → train/val.jsonl
python -m training.teacher --collect --val-frac 0.28
# 4) LoRA finetune
sbatch training/slurm/launch_sft.sh google/gemma-4-12B-it 3 v5
# 5) evaluate base vs finetuned
sbatch training/slurm/launch_eval.sh google/gemma-4-12B-it v5 201
```
The teacher only writes prose; the engine owns every number and discards any instance
that doesn't compute to a viable DPR (PAT ≥ 0, balanced, avg DSCR ≥ 1.5).

The SLURM scripts carry no site-specific paths — submit them from the repo root and they
resolve everything relative to themselves. Override per cluster with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | model/dataset cache (point at fast local disk) |
| `DPR_VENV` | `venv` | virtualenv to activate |
| `DPR_MAX_GPU_INDEX` | `99` | only auto-select GPUs at or below this index |
| `EVAL_GPU` | *(auto)* | pin `launch_eval.sh` to one GPU |

Partition, node and GPU count come from the `#SBATCH` headers; override on the command
line, e.g. `sbatch --partition=a100 --gres=gpu:2 training/slurm/launch_sft.sh …`.

## Quick checks (no GPU)
```bash
python -m dpr.engine.financial_engine            # engine self-test vs NIFTEM ghee ground truth
python -m training.gpt_gen_payloads --dry-run --per-line 3   # end-to-end dataset dry-run (no API)
```

## Docs
- **How a request flows** → [docs/flow.md](docs/flow.md)
- **What each file does** → [docs/FILE_FLOW.md](docs/FILE_FLOW.md)
- **Every field/param & where its value comes from** → [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
- **Financial formulas** → [docs/FINANCIAL_ENGINE.md](docs/FINANCIAL_ENGINE.md)
- **Running the HTTP service** → [docs/serving.md](docs/serving.md)

## The 12 lines & schemes
Rice / Dhal / Flour Mill · Oil Extraction *(any oilseed)* · Ghee · Spice · Juice *(any
fruit)* · Fruit & Veg Grading · Nursery Hardening · Vermicompost · Custom Hiring Centre ·
Storage / Warehouse. Schemes: PM-FME (35 % subsidy), MIDH (40 %), AIF (3 % interest
subvention), self-funded; unknown/"Others" → self-funded + a note.


