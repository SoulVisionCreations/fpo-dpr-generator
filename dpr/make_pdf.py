#!/usr/bin/env python3
"""
make_pdf.py — package the 8 best finetuned-gemma DPRs (one per business
line) as client-validation PDFs, each paired with the exact JSON input that produced
it. Rendered with the project's pandoc + tectonic toolchain (booktabs tables, running
header) using pdf/dpr_pdf.yaml, with the Table of Contents moved to AFTER the Project
Summary section.

  ./venv/bin/python make_pdf.py
Output: output/client_samples/<NN>_<line>__<id>.pdf  +  ....input.json
"""
from __future__ import annotations
import json, os, re, shutil, subprocess
from pathlib import Path

from dpr import config
EVAL = config.OUTPUT_DIR / "ft_eval"
INST = config.FINETUNE_DIR / "instances"
OUT  = config.OUTPUT_DIR / "client_samples"
SRC  = OUT / "_md"                                   # preprocessed markdown (kept for audit)
MODEL_TAG = "google_gemma-4-12B-it__finetuned"

PANDOC   = config.PANDOC
TECTONIC = config.TECTONIC
STYLE    = config.PDF_STYLE
NOTOC    = config.PDF_NOTOC

# best clean finetuned DPR per business line (max §2, assemble-ok, 0 number-issues).
# The name is the client-facing PDF filename (business line only, nothing else).
SELECTED = [
    ("Rice Mill",                    "rice_mill__g019"),
    ("Mustard Oil Extraction",       "mustard_oil__g028"),
    ("Amla Juice Processing",        "amla__g033"),
    ("Nursery Hardening",            "nursery_hardening__g037"),
    ("Vermicompost",                 "vermicompost__g038"),
    ("Fruit and Vegetable Grading",  "veg_fruit_grading__g022"),
    ("Custom Hiring Centre",         "custom_hiring_centre__g036"),
    ("Warehouse and Cold Storage",   "warehouse_cold_storage__g025"),
]

# enlarge the Project Summary (bigger font + roomier rows) so page 1 fills nicely;
# the group is closed just before the TOC so the rest of the DPR keeps normal size.
SUMMARY_OPEN = ("\n```{=latex}\n\\begingroup\n\\vspace*{2.5em}\n"
                "\\setlength{\\tabcolsep}{12pt}\n\\renewcommand{\\arraystretch}{2.6}\n"
                "\\large\n```\n")
CLOSE_AND_TOC = ("\n```{=latex}\n\\endgroup\n\\clearpage\n\\tableofcontents\n"
                 "\\clearpage\n```\n")
TOC_PLAIN = "\n```{=latex}\n\\clearpage\n\\tableofcontents\n\\clearpage\n```\n"


def normalize_tables(md: str) -> str:
    """pandoc pipe_tables need a BLANK LINE before and after the table block. The
    finetuned model often writes its commentary sentence directly above the injected
    table with no blank line, so pandoc renders the pipes as literal text. Insert the
    missing blank lines around every run of '|...' rows."""
    def is_row(s: str) -> bool:
        return s.lstrip().startswith("|")
    out: list[str] = []
    for i, l in enumerate(md.split("\n")):
        prev = out[-1] if out else ""
        if is_row(l) and prev.strip() and not is_row(prev):      # blank line BEFORE table
            out.append("")
        out.append(l)
    fixed: list[str] = []
    lines = out
    for i, l in enumerate(lines):
        fixed.append(l)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if is_row(l) and nxt.strip() and not is_row(nxt):        # blank line AFTER table
            fixed.append("")
    return "\n".join(fixed)


def _texesc(s: str) -> str:
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                 ("_", "\\_"), ("#", "\\#"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}")]:
        s = s.replace(a, b)
    return s


def rebuild_process_flow(md: str) -> str:
    """Replace the '## 4. Process Flow Chart' ASCII '|/v' steps with a real vertical
    TikZ flow chart (rounded boxes joined by downward arrows)."""
    lines = md.split("\n")
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"^##\s+4\.\s+Process Flow", l)), None)
    if start is None:
        return md
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
               len(lines))
    steps = [m.group(1).strip() for l in lines[start + 1:end]
             if (m := re.match(r"^\s*\d+\.\s+(.*\S)", l))]
    if len(steps) < 2:
        return md
    tk = ["", "```{=latex}", "\\begin{center}",
          "\\begin{tikzpicture}[node distance=5mm]"]
    for i, s in enumerate(steps):
        pos = "" if i == 0 else f",below=of n{i-1}"
        tk.append(f"\\node[flowbox{pos}] (n{i}) {{{_texesc(s)}}};")
    for i in range(1, len(steps)):
        tk.append(f"\\draw[flowarrow] (n{i-1}) -- (n{i});")
    tk += ["\\end{tikzpicture}", "\\end{center}", "```", ""]
    return "\n".join(lines[:start + 1] + tk + lines[end:])


def inject_toc_after_summary(md: str) -> str:
    """Enlarge the Project Summary block and place \\tableofcontents right after it
    (title is the H1; Project Summary is the 1st ##, Product/Market is the 2nd ##)."""
    lines = md.split("\n")
    h2 = [i for i, l in enumerate(lines) if l.startswith("## ")]
    if len(h2) >= 2:
        lines.insert(h2[1], CLOSE_AND_TOC)           # close big-font group + TOC
        lines.insert(h2[0], SUMMARY_OPEN)            # open big-font before Project Summary
    else:                                            # fallback: after the H1 title
        h1 = next((i for i, l in enumerate(lines) if l.startswith("# ")), 0)
        lines.insert(h1 + 1, TOC_PLAIN)
    return "\n".join(lines)


def main() -> int:
    for d in (OUT,):
        if d.exists():
            shutil.rmtree(d)
    SRC.mkdir(parents=True)

    # tectonic writes its bundle cache under HOME; keep it inside the project.
    env = dict(os.environ)
    env["HOME"] = str(BASE / ".tectonic-home")
    env["XDG_CACHE_HOME"] = str(BASE / ".tectonic-home" / ".cache")

    ok_all = True
    for name, iid in SELECTED:
        md_path = EVAL / f"{iid}__{MODEL_TAG}.md"
        inst_path = INST / f"{iid}.json"
        assert md_path.exists() and inst_path.exists(), f"missing source for {iid}"

        # preprocess markdown (enlarge summary + TOC after it) and render
        src_md = SRC / f"{name}.md"
        src_md.write_text(inject_toc_after_summary(
            rebuild_process_flow(normalize_tables(md_path.read_text(encoding="utf-8")))))
        pdf_out = OUT / f"{name}.pdf"
        cmd = [str(PANDOC), str(src_md), "--defaults", str(STYLE),
               "--defaults", str(NOTOC), "--pdf-engine", str(TECTONIC), "-o", str(pdf_out)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ok = r.returncode == 0 and pdf_out.exists()
        ok_all &= ok

        # pair the exact engine instance JSON
        (OUT / f"{name}.input.json").write_text(
            json.dumps(json.loads(inst_path.read_text()), ensure_ascii=False, indent=2))

        size = f"{pdf_out.stat().st_size//1024}KB" if ok else "FAILED"
        print(f"  {name:32} {'ok' if ok else 'ERR'} {size}")
        if not ok:
            print("    " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:200])

    print(f"\n[{'ok' if ok_all else 'PARTIAL'}] PDFs + input JSONs -> {OUT}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
