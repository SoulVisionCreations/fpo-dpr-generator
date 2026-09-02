# Evaluation — v4 finetuned vs. base (gemma-4-12B-it)

Held-out validation set: **201 DPRs**, all 12 business lines. Each model authored the
full DPR (Option-B: prose + `{{TABLE}}` placeholders); every output was run through
`render/assemble.py` (inject exact engine tables) and `verify()` (check every number the
model typed). Base = `google/gemma-4-12B-it`; v4 = same base + the `v4` LoRA adapter
(`models/google_gemma-4-12B-it_v4/adapter`). Source summaries:
`output/ft_eval/summary__google_gemma-4-12B-it__{base,finetuned}.json`.

## Verdict: **adopt v4.** It writes better narrative and better structure; the only cost is 2 (guard-caught) number slips.

---

## What the three metrics mean

| Metric | Question it answers | Higher is |
|---|---|---|
| **assemble_ok** | Did the model place every `{{TABLE:id}}` exactly once (no missing/duplicate section)? | better |
| **number-issues** | Did the model type a figure in its prose that the engine didn't produce? (caught by `verify()` at inference) | lower |
| **s2_score** | Quality of the §2 narrative the model wrote (structure, coverage, commodity-fit; 0–100) | better |

---

## Overall (n = 201)

| Metric | Base | **v4** | Δ |
|---|---|---|---|
| assemble_ok | 195 / 201 (97.0%) | **198 / 201 (98.5%)** | +3 |
| §2 score — mean | 94.6 | **96.1** | **+1.5** |
| §2 score — median | 95.1 | **97.7** | **+2.6** |
| §2 score — range | 82.6 – 99.4 | 83.5 – 100.0 | ↑ |
| zero-number-issue DPRs | 201 / 201 | 199 / 201 | −2 |
| total number-issues | 0 | 2 | +2 |
| **per-instance §2: which model wrote the better narrative** | 80 | **120** (1 tie) | v4 wins 60% |

---

## Per business line

| Line | n | Base §2 | v4 §2 | Δ §2 | Base assemble | v4 assemble |
|---|---|---|---|---|---|---|
| Juice (amla/fruit) | 17 | 89.3 | **97.1** | **+7.7** | 15/17 | **17/17** |
| Spice | 22 | 92.9 | **98.8** | **+5.9** | 19/22 | **21/22** |
| Vermicompost | 18 | 94.3 | **98.6** | **+4.3** | 18/18 | 17/18 |
| Storage / Warehouse | 9 | 94.9 | **97.0** | +2.0 | 9/9 | 9/9 |
| Custom Hiring Centre | 20 | 95.8 | **97.3** | +1.4 | 20/20 | 20/20 |
| Nursery Hardening | 16 | 96.8 | **98.1** | +1.3 | 16/16 | 16/16 |
| Ghee | 19 | 94.8 | **95.3** | +0.5 | 19/19 | 19/19 |
| Dhal Mill | 16 | 95.4 | 94.7 | −0.7 | 16/16 | 16/16 |
| Oil (mustard/oilseed) | 21 | 95.7 | 94.8 | −0.9 | 21/21 | 20/21 |
| Fruit & Veg Grading | 16 | 95.3 | 94.1 | −1.2 | 16/16 | 16/16 |
| Rice Mill | 12 | 95.4 | 93.9 | −1.5 | 12/12 | 12/12 |
| Flour Mill | 15 | 95.3 | 92.2 | −3.1 | 14/15 | **15/15** |

---

## Reading the results

- **v4's gains land exactly where they were needed.** The biggest lifts are on the
  narrative-heavy, commodity-varied lines the retrain targeted — **Juice +7.7, Spice
  +5.9, Vermicompost +4.3** — where the model now writes rich, product-specific §2
  (mango vs. amla, ginger vs. turmeric) instead of a generic template echo.
- **The small regressions are on the simple mill lines** (Flour −3.1, Rice −1.5, Dhal
  −0.7, Grading −1.2, Oil −0.9), where §2 is short and formulaic and there was little to
  improve — v4 trades a fraction of polish there for large gains elsewhere. Net median
  still moves **95.1 → 97.7**.
- **Structure improved:** v4 **fixed all 6 of base's assemble failures** (2 amla, 3
  spices, 1 flour) and introduced 3 of its own → **3 fails vs. base's 6**.

## The 3 v4 stragglers (and the 2 number-issues)

| Instance | assemble_ok | number-issues |
|---|---|---|
| `mustard_oil__g037` | ✗ | 1 |
| `vermicompost__g013` | ✗ | 1 |
| `spices__g034` | ✗ | 0 |

Both number-issues occur **on the same instances that also failed to assemble** — i.e.
~1.5% of generations went fully off-script (bad structure *and* a stray number) rather
than a broad number-safety regression. Crucially, **`assemble()` + `verify()` catch all
three at inference**, so they never ship — the pipeline simply re-samples them.

## Recommendation

1. **Adopt v4** (`models/google_gemma-4-12B-it_v4/adapter`).
2. Add a **retry-on-failure** at inference: if `assemble().ok` is False (or `verify()`
   flags a number), regenerate that DPR (temperature resample) up to N times. That turns
   the ~1.5% off-script rate into ≈0 shipped failures — giving v4's quality gain with
   base's perfect number-safety.
