# Quickstart — run the DPR endpoint

A GPU box with **≥ 40 GB** free VRAM (verified on an L40S). One POST in, a finished
DPR in markdown out, in about 70 seconds.

## 1. Get the adapter

The LoRA adapter (112 MB) is not in this repo:

```bash
git clone <this repo> && cd fpo-dpr-generator

mkdir -p models/google_gemma-4-12B-it_v4/adapter
aws s3 sync \
  s3://fashion-simulation/developers/adityas/indiaai/google_gemma-4-12B-it_v4/adapter/ \
  models/google_gemma-4-12B-it_v4/adapter/
```

The base model (`google/gemma-4-12B-it`, 24 GB) downloads itself from Hugging Face
on first start — it is public, no token needed.

## 2. Run it

**Directly** (no root required):

```bash
python3 -m venv venv && . venv/bin/activate
pip install -e ".[model,serve]" vllm==0.30.0

export DPR_API_KEY='choose-a-secret'
python -m uvicorn serve.app:app --host 0.0.0.0 --port 8000 --workers 1 \
    --timeout-keep-alive 305
```

**Or with Docker** (needs root; the image already has CUDA/torch/vLLM matched):

```bash
docker build -t dpr-api .
docker run -d --name dpr-api --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 -e DPR_API_KEY='choose-a-secret' \
  -v $HOME/.cache/huggingface:/models/hf \
  -v $PWD/models/google_gemma-4-12B-it_v4/adapter:/adapters/v4:ro \
  dpr-api
```

`--ipc=host` is required: vLLM shares memory between its API and engine processes,
and Docker's default 64 MB `/dev/shm` makes it crash obscurely.

First start downloads 24 GB (~20 min); later starts take 2–4 minutes. Use `tmux` if
running directly and you want it to survive disconnecting.

## 3. Test it

```bash
curl -s localhost:8000/ready     # {"state":"ready"} once the model has loaded

curl -s -X POST localhost:8000/v1/dpr:generate \
  -H "X-API-Key: $DPR_API_KEY" -H 'content-type: application/json' \
  -d "$(python -c "import json;print(json.dumps({'payload':json.load(open('data/examples/mango_juice.json'))}))")" \
  | python -c "import json,sys; r=json.load(sys.stdin); print(r['meta']); open('out.md','w').write(r['markdown'])"
```

The engine is deterministic, so this payload must produce exactly:

```
total_project_cost_lakh : 68.54
pat_year1               : 24.61
average_dscr            : 8.65
verification.ok         : true
```

If those match, the install is correct. Swagger UI is at `http://<host>:8000/docs`.

## Endpoints

```
POST /v1/dpr:generate   needs X-API-Key   →  the service
GET  /health            open              →  liveness  (always 200)
GET  /ready             open              →  readiness (200 once loaded)
GET  /v1/info           needs X-API-Key   →  model, adapter, limits
```

Set `DPR_ENABLE_DOCS=0` to remove `/docs`, `/redoc` and `/openapi.json` in
production. Keep `/health` and `/ready` — health checks cannot send an API key, and
they expose only the engine's lifecycle state.

## Request format

```json
{"payload": { ...the website form JSON, exactly as the form produces it... }}
```

Use [`data/examples/mango_juice.json`](data/examples/mango_juice.json) as a template
— wrap it in `{"payload": ...}` and edit the fields. Everything is optional; the
engine fills line defaults and lists what it defaulted in the response's `warnings`.

Two things to get right: `businessLine` must match one of the 12 known lines, and
`salePrice` is **Rs per capacity unit** (with `capacity` in KL/day, that means Rs per
KL — not per litre).

The response carries `markdown`, `verification` (did every number check out?), `meta`
(headline figures, token counts) and `warnings`.

---

Full reference — every environment variable, tuning and memory sizing notes:
[docs/serving.md](docs/serving.md).
