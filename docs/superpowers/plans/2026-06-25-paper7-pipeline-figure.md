# Paper 7 CEUS Pipeline Figure Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the embedded CEUS pipeline placeholder with a standalone vector figure and wire that asset into both manuscript copies.

**Architecture:** Generate one publication-quality pipeline diagram as a reusable PDF/PNG pair under `submission/ceus/05_figures/`. Update both CEUS manuscripts to include the figure file from the sibling figure directory instead of the inline text box. Keep the caption and claim boundary unchanged so the manuscript text remains stable.

**Tech Stack:** Python, Matplotlib, LaTeX, pdflatex.

---

### Task 1: Generate the standalone pipeline figure

**Files:**
- Create: `paper7/make_ceus_pipeline_figure.py`
- Create: `submission/ceus/05_figures/figure_1_pipeline.pdf`
- Create: `submission/ceus/05_figures/figure_1_pipeline.png`

- [ ] **Step 1: Add the figure generator**

Create a small Matplotlib script that draws a single-row pipeline diagram with five labeled stages and arrows:
`Real county environment -> trajectory collection -> learned transition model -> learned county environment -> policy optimization with optional observational reward calibration -> real-environment evaluation`.

- [ ] **Step 2: Run the script**

Run: `python paper7/make_ceus_pipeline_figure.py`

Expected: both `figure_1_pipeline.pdf` and `figure_1_pipeline.png` exist in `submission/ceus/05_figures/`.

- [ ] **Step 3: Sanity-check the asset**

Open the generated files or inspect their size to confirm the figure is non-empty, landscape-oriented, and readable at manuscript width.

- [ ] **Step 4: Commit**

Run:

```powershell
git add paper7/make_ceus_pipeline_figure.py submission/ceus/05_figures/figure_1_pipeline.pdf submission/ceus/05_figures/figure_1_pipeline.png
git commit -m "feat: add standalone CEUS pipeline figure"
```

### Task 2: Replace the embedded placeholder in both manuscripts

**Files:**
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex`
- Modify: `submission/ceus/06_latex_source_editable/manuscript_signed.tex`

- [ ] **Step 1: Swap the inline box for `\includegraphics`**

Replace the current `\fbox`/`minipage` pipeline block with a figure that includes:
`\includegraphics[width=0.92\textwidth]{../05_figures/figure_1_pipeline.pdf}`.

- [ ] **Step 2: Keep the caption and label stable**

Preserve `\caption{Pipeline for learned-environment policy training. ...}` and `\label{fig:pipeline}` so the surrounding prose does not need rewriting.

- [ ] **Step 3: Commit**

Run:

```powershell
git add submission/ceus/01_main_document_anonymous/manuscript.tex submission/ceus/06_latex_source_editable/manuscript_signed.tex
git commit -m "docs: replace CEUS pipeline placeholder with standalone figure"
```

### Task 3: Verify manuscript builds and package notes

**Files:**
- Modify: `submission/ceus/05_figures/README_figures.md`
- Modify: `submission/ceus/README_CEUS_submission_package.md`

- [ ] **Step 1: Update figure notes**

Revise the figure README to state that `figure_1_pipeline.pdf/png` now exist and that the remaining recommended figures are still optional.

- [ ] **Step 2: Refresh package notes if needed**

Adjust the package README only if it still says the pipeline figure is missing.

- [ ] **Step 3: Compile both manuscripts**

Run `pdflatex -interaction=nonstopmode manuscript.tex` twice in each of:
`submission/ceus/01_main_document_anonymous`
and `submission/ceus/06_latex_source_editable`.

- [ ] **Step 4: Check logs**

Run:

```powershell
rg -n "undefined|Undefined|Citation|Reference|Fatal|Emergency|Error|Rerun" manuscript.log
```

Expected: no fatal errors or unresolved references.

- [ ] **Step 5: Final commit**

Run:

```powershell
git add submission/ceus/05_figures/README_figures.md submission/ceus/README_CEUS_submission_package.md
git commit -m "docs: refresh CEUS figure package notes"
```

### Task 4: Final verification

- [ ] **Step 1: Check git status**

Run: `git status -sb`

Expected: only intentional figure, manuscript, and note updates are present.

- [ ] **Step 2: Confirm the figure file is referenced**

Run:

```powershell
rg -n "figure_1_pipeline|fig:pipeline" submission\ceus\01_main_document_anonymous\manuscript.tex submission\ceus\06_latex_source_editable\manuscript_signed.tex
```

Expected: both manuscripts reference the standalone figure file.
