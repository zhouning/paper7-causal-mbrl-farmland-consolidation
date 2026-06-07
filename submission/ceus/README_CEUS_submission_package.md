# CEUS Submission Package - Paper 7

Target journal: Computers, Environment and Urban Systems.

Manuscript title: A Causally Calibrated Learned Environment for Reinforcement Learning-Based Farmland Consolidation Planning.

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

Signed source copy and anonymous source copy for editable-source upload or later editorial use.

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

- The source manuscript was retitled for CEUS positioning.
- The abstract was shortened to satisfy the CEUS 250-word limit.
- The anonymous manuscript uses manual `thebibliography` references, so BibTeX is not required.
- The body no longer relies on four unpublished author self-citations; those bibliography entries were removed.
- The manuscript defines `Fig.~\ref{fig:pipeline}` as an embedded placeholder pipeline figure. A standalone publication-quality pipeline figure is still recommended before final CEUS submission.

## Build and Verification Snapshot

Checked on 2026-06-07.

- Anonymous manuscript compiled successfully: `01_main_document_anonymous/manuscript.pdf`, 16 pages.
- Signed source compiled successfully: `06_latex_source_editable/manuscript_signed.pdf`, 16 pages.
- Title page compiled successfully: `02_title_page_author_info/title_page.pdf`, 1 page.
- Abstract length after CEUS retargeting and review-driven revision: 198 words.
- Highlights: five bullets; each is under 85 characters including the bullet marker.
- Anonymous source grep: no exact hits for `Ning`, `Zhou`, `Xiang`, `Jing`, `Peking`, `pku`, `jingxiang`, or `School of Software`.
- Self-citation grep: no hits for `zhou2026`, `Author(s)`, `Anonymized self-citation`, or related unpublished-self-citation markers.
- LaTeX log scan: no undefined citations, undefined references, fatal errors, or emergency stops.
- Anonymous source zip prepared: `CEUS_paper7_latex_source_anonymous.zip`.
- Remaining nonblocking issue: only minor overfull `hbox` warnings remain; no blocking LaTeX errors were found.
