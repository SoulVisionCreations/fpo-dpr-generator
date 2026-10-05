# Bundled image — build, publish, run

A single image with the model baked in. The client pulls it and runs one command:
no Hugging Face access, no mounted weights, no adapter to place.

```
~37 GB image  =  ~24 GB bf16 merged weights  +  ~12 GB vLLM base  +  ~1 MB app
```

---

## GPU requirement

bf16 weights are ~24 GB, so the card needs headroom for weights plus KV cache.

| GPU | VRAM | KV cache at 0.90 util | Runs? |
|---|---:|---:|---|
| L40S | 44.4 GB | ~13 GB | **yes** (verified) |
| A100-40 | 40 GB | ~9 GB | yes |
| A100-80 / H100 | 80 GB | ~45 GB | yes |
| L4 / A10G / RTX 4090 | 24 GB | — | **no** |

A 24 GB card needs FP8 weights (~12 GB). That is a one-line change to this
Dockerfile plus a verification run; it is not built today.

The host needs an NVIDIA driver and `nvidia-container-toolkit`. Nothing else —
CUDA, torch and vLLM all live inside the image.

---

## 1. Merge the adapter

Run on a machine with ~50 GB free disk and ~30 GB RAM. **Not a laptop** — the
base model alone is 24 GB. The L40S box is the natural place; merging itself is
CPU work, the GPU is not used.

```bash
cd ~/fpo_dpr/fpo-dpr-generator
python scripts/merge_adapter.py --out models/merged --verify
```

Takes 10–30 minutes, mostly loading and writing 24 GB. `--verify` reloads the
result and checks the chat template applies and the stop tokens survived.

Expected output ends with roughly:

```
[done] 23.9x GB in NNNs
       model-00001-of-0000X.safetensors   x.xx GB
       tokenizer.json                     0.03 GB
       chat_template.jinja                0.00 GB
       generation_config.json             0.00 GB
```

`generation_config.json` matters: it carries `eos_token_id = [1, 106, 50]`, the
list that makes the model stop on `<turn|>`. The server passes its own copy too,
so this is belt and braces.

## 2. Build

```bash
docker build -f Dockerfile.bundled -t ghcr.io/<org>/dpr-api:v4-bf16 .
```

The build fails fast if the weights or tokenizer are missing, and imports the app
to catch a broken module before the image is published.

Expect 15–40 minutes, almost all of it copying 24 GB into a layer. Later builds
after a **code** change reuse that layer and take seconds — the weights are
copied before the application code precisely so that holds.

## 3. Publish to GHCR

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-username> --password-stdin
docker push ghcr.io/<org>/dpr-api:v4-bf16
```

The token needs `write:packages`. A first push creates the package **private**;
make it public from the package page on GitHub if the client should pull without
credentials.

Tag deliberately — `v4-bf16` names the adapter version and the precision, so a
later FP8 build or a v5 adapter does not overwrite it. Avoid `latest` for
something this large and this specific.

## 4. Running it

The agent listens on **port 9000** and serves the generation endpoint at
**`/dpr`**, so a gateway can be pointed straight at
`http://dpr-agent:9000/dpr`.

### Behind the UAI connector

```bash
docker network create uai        # once

docker run -d --name dpr-agent --network uai --restart unless-stopped \
  --gpus all --ipc=host --shm-size=8g \
  -e DPR_API_KEY='<shared-secret>' \
  ghcr.io/soulvisioncreations/dpr-agent:v4-bf16

docker run -d --name uai-connector --network uai --restart unless-stopped \
  -p 80:80 -p 443:443 -p 127.0.0.1:7000:7000 \
  -v uai-connector:/data \
  -e UAI_HOST=uai.nafpo.in \
  -e UAI_CONTROL_PLANE=https://staging.unifiedai.org.in \
  -e UAI_SCOPE=nafpo.dpr \
  -e UAI_AGENT_URL=http://dpr-agent:9000/dpr \
  ghcr.io/soulvisioncreations/uai-connector:<tag>
```

The agent **publishes no ports**. The connector reaches it by container name on
the shared `uai` network and terminates TLS at the edge. Never add `-p` to the
agent unless you intend it to be reachable directly.

`--ipc=host` is required, not cosmetic: vLLM shares memory between its API and
engine processes, and Docker's default 64 MB `/dev/shm` makes it crash in ways
that are hard to read.

### Standalone (no connector)

```bash
docker run -d --name dpr-agent --gpus all --ipc=host --shm-size=8g \
  -p 9000:9000 --restart unless-stopped \
  -e DPR_API_KEY='<secret>' \
  ghcr.io/soulvisioncreations/dpr-agent:v4-bf16
```

If the package is private, authenticate first:

```bash
echo <token> | docker login ghcr.io -u <username> --password-stdin
```

### Request format

`/dpr` accepts the form payload **with or without** the envelope, so a gateway
that forwards the caller's body unchanged works either way:

```jsonc
{"document_type": "dpr", "fields": { ... }}              // bare form
{"payload": {"document_type": "dpr", "fields": {...}}}   // envelope
{"payload": {...}, "options": {"require_complete_inputs": true}}
```

Both produce identical output. `options` is only reachable through the envelope.

### Check it is up

```bash
docker exec dpr-agent curl -s localhost:9000/ready   # {"state":"ready"}

docker exec dpr-agent curl -s -X POST localhost:9000/dpr \
  -H "X-API-Key: $DPR_API_KEY" -H 'content-type: application/json' \
  -d '{"document_type":"dpr","fields":{"businessLine":"Rice Mill"}}'
```

Startup is 2–5 minutes — weight load and CUDA graph capture, no download.

### Authentication

The image fails closed: without `DPR_API_KEY` it refuses to start. Behind the
connector on a private network you may set `DPR_REQUIRE_AUTH=0` instead, but
only when the agent publishes no ports.

---

## Tuning per card

All settable at `docker run` with `-e`, no rebuild:

| Variable | Default | When to change |
|---|---|---|
| `DPR_MAX_MODEL_LEN` | 8192 | Lower to 6144 first if KV cache is tight |
| `DPR_GPU_MEMORY_UTILIZATION` | 0.90 | Never above 0.93 on a headless card |
| `DPR_MAX_NUM_SEQS` | 8 | Lower on a 40 GB card |
| `DPR_MAX_CONCURRENCY` | 4 | In-flight generations; excess gets 429 |
| `DPR_REQUEST_TIMEOUT_S` | 240 | Per-request deadline |
| `DPR_ENABLE_DOCS` | 1 | `0` removes `/docs` and `/openapi.json` |

Read vLLM's `GPU KV cache size: N tokens` line from `docker logs` and tune
against that rather than the estimates above.

---

## How this differs from `./Dockerfile`

| | `Dockerfile` | `Dockerfile.bundled` |
|---|---|---|
| Weights | downloaded from HF at first run | baked in |
| Adapter | mounted from the host | merged into the weights |
| Image size | ~12 GB | ~37 GB |
| Volumes needed | two | none |
| Network at runtime | yes, first boot | **none** |
| LoRA at runtime | yes | no — already merged |

Keep both. The mounted variant is better for iterating on adapter versions; the
bundled one is what ships to a client.

---

## Two things that will bite

**`.dockerignore` excludes `models/`.** The bundled build needs
`models/merged/`, so it uses its own `Dockerfile.bundled.dockerignore`, which
re-includes that one directory. BuildKit picks this up automatically from the
filename. If a build fails with "merged weights missing", check that file is
present and that BuildKit is enabled (`DOCKER_BUILDKIT=1`).

**`PYTHONOPTIMIZE` must stay `0`.** The engine's balance check — the only
guarantee that means of finance equals total project cost — is a Python
`assert`, and `python -O` strips it silently. The image pins it to `0` and
`serve/app.py` refuses to start otherwise.
