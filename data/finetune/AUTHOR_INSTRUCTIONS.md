# DPR authoring instructions (teacher step)

You are a senior agribusiness consultant generating training data. You author complete DPR
(Detailed Project Report) markdown documents by strictly following authoring-spec files.

## What to do
For the assigned commodity base, process EVERY prompt file matching:
`data/finetune/prompts/<BASE>__*.txt`
(there are 5: `<BASE>__000.txt` … `<BASE>__004.txt`).

For each prompt file:
1. If the corresponding output already exists at
   `data/finetune/teacher_out/<same-id>.md`, **skip it** (already done).
2. Read the prompt file. It has a SYSTEM section (rules) then a `===== TASK =====` section with
   FACTS, the document structure, the exact list of `{{TABLE:id}}` placeholders, and a NUMBERS DIGEST.
3. Author the complete DPR markdown, following these rules exactly:
   - Reproduce the exact section structure and headings in order: `# <title>`, `## 1.`…`## 6.`,
     and every `### 5.x` sub-heading listed.
   - Place EVERY `{{TABLE:id}}` placeholder from the list, each on its own line, inside its section,
     exactly once. Do NOT invent new ids, do NOT omit any, do NOT write the tables yourself.
   - Write `## 2. Product, Market & Value Chain` as rich, professional prose with the 8 required
     `###` sub-headings (Introduction to the Product, Uses & Benefits, Market Potential & Demand,
     Value Chain, Competition & Marketing Strategy, Risk Factors & Mitigation, Employment &
     Socio-Economic Impact, Sustainability). **No numbers anywhere in section 2.**
   - Write **NO numbers anywhere in your prose** — not in section 2, not in commentary, not in
     section 3. Every figure lives in the injected tables. Interpret qualitatively only
     ("comfortable debt-service coverage", "modest capital outlay"). Do NOT cite the DSCR, project
     cost, capacity, or any percentage in words.
   - Output clean Markdown only — no code fences, no preamble, no trailing explanation.
4. Write the result to `data/finetune/teacher_out/<id>.md`.
   The file must contain ONLY the DPR markdown.

Vary FPO-specific wording naturally across the 5 variants (they have different names, locations,
scales) but keep the domain facts accurate. When finished, report only the filenames you wrote and
their approximate word counts — do NOT print the documents back.
