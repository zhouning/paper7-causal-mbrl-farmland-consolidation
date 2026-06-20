# CEUS Submission Package - Paper 7

Target journal: Computers, Environment and Urban Systems.

Manuscript title: A Treatment-Effect-Informed Learned Environment for Reinforcement Learning-Based Farmland Consolidation Planning.

## Directory Structure

`01_main_document_anonymous/`

Double-anonymized LaTeX manuscript for the main manuscript upload slot. Author names, affiliation, email, and author-identifying self-citation details have been removed or anonymized.

`02_title_page_author_info/`

Separate title page with author names, affiliation, corresponding-author email, funding note, declaration of interest, CRediT, data availability, and generative-AI statement.

`03_highlights/`

Highlights file with five short bullet points.

`04_cover_letter/`

Editor-facing cover letter tailored to CEUS.

`05_figures/`

Figure staging folder. No standalone figure files were present in the source `paper7` directory. The manuscript now includes an embedded text-box pipeline figure; see `README_figures.md` for recommended publication-quality figure work.

`06_latex_source_editable/`

Signed source copy and anonymous source copy for editable-source upload or later editorial use. Use `CEUS_paper7_latex_source_anonymous.zip` for the anonymous editable-source upload.

`07_declarations_and_checks/`

Standalone declaration drafts and submission checklist.

`08_supplementary_optional/`

Small supplementary result summaries copied from the paper7 result folders.

`99_admin_notes/`

Original English and Chinese Word files plus the original LaTeX source. Do not upload unless needed for internal reference.

## Suggested Editorial Manager Upload Mapping

1. Cover Letter: `04_cover_letter/cover_letter.txt`
2. Highlights: `03_highlights/highlights.txt`
3. Title Page / Author Information: `02_title_page_author_info/title_page.pdf` or `title_page.tex`
4. Manuscript: `01_main_document_anonymous/manuscript.pdf`
5. Editable Source: `CEUS_paper7_latex_source_anonymous.zip`
6. Figures: files in `05_figures/`, after the missing figures are prepared
7. CRediT Author Statement: `07_declarations_and_checks/credit_statement.txt`
8. Declaration of Interest: `07_declarations_and_checks/declaration_of_interest.txt`
9. Data Availability: enter the statement from `07_declarations_and_checks/data_availability_statement.txt`
10. Supplementary Material: files in `08_supplementary_optional/`, only if desired

## Notes

- The source manuscript was retitled and revised for CEUS positioning.
- The abstract was shortened to satisfy the CEUS 250-word limit after the added experimental evidence.
- The anonymous manuscript uses manual `thebibliography` references, so BibTeX is not required.
- The body no longer relies on four unpublished author self-citations; those bibliography entries were removed.
- The manuscript defines `Fig.~\ref{fig:pipeline}` as an embedded placeholder pipeline figure. A standalone publication-quality pipeline figure is still recommended before final CEUS submission.

## Build and Verification Snapshot

Anonymous manuscript and evidence-chain refresh checked on 2026-06-20.

- Anonymous manuscript compiled successfully after two final `pdflatex` passes: `01_main_document_anonymous/manuscript.pdf`, 29 pages.
- Abstract length after CEUS retargeting and experimental-strengthening revision: 234 words by repository preflight count.
- Policy-induced learned-vs-real diagnostic expanded to 15 calibrated policy seeds and validated in `paper7/results/revision/policy_induced_diagnostics_15seed.json`.
- New CEUS strengthening artifacts include `reward_scaling_comparator.json`, `planning_significance_audit.json`, expanded `transition_rollout_diagnostics.json`, `dongxing_rl_lite.json`, `dongxing_multistep_mbrl_policy.json`, and the Dongxing full-rigor result files under `paper7/results/full_rigor/`.
- New robustness artifact `trajectory_source_ablation.json` compares random-only, greedy-only, and mixed trajectory sources for the learned transition model.
- End-to-end evidence audit refreshed in `paper7/results/revision/end_to_end_validation.json`; the Bishan learned-policy chain is supported and Dongxing is bounded as an external full-reward local counterpart, not direct Bishan-to-Dongxing policy transfer.
- The Dongxing counterpart now includes one-step model-based action selection, held-out scoring optimization, and a multi-step learned-environment policy as a structural counterpart to the Bishan learned-environment loop.
- Verification command `python -m paper7.end_to_end_validation --out paper7/results/revision/end_to_end_validation.json` returned `supported_with_bounded_external_scope`.
- Targeted test command `python -m pytest tests/test_end_to_end_validation.py tests/test_trajectory_source_ablation.py -q` returned exit code 0 with `33 passed`; pytest also reported one nonblocking Windows cache warning.
- Highlights: five bullets; each is under 85 characters including the bullet marker.
- Anonymous source grep: no exact hits for `Ning`, `Zhou`, `Xiang`, `Jing`, `Peking`, `pku`, `jingxiang`, or `School of Software`.
- Self-citation grep: no hits for `zhou2026`, `Author(s)`, `Anonymized self-citation`, or related unpublished-self-citation markers.
- LaTeX log scan: no undefined citations, undefined references, fatal errors, or emergency stops.
- Anonymous source copy synchronized with the main anonymous manuscript, and anonymous source zip prepared: `CEUS_paper7_latex_source_anonymous.zip`.
- Title page PDF compiled successfully: `02_title_page_author_info/title_page.pdf`, 1 page.
- Signed manuscript compiled successfully after two `pdflatex` passes: `06_latex_source_editable/manuscript_signed.pdf`, 29 pages.
- Title page, cover letter, highlights, signed source, and anonymous source were synchronized with the treatment-effect-informed wording and the added experiment boundary.
- Remaining nonblocking issue: overfull/underfull `hbox` warnings remain for long technical phrases, paths, and compact tables; no blocking LaTeX errors were found.
