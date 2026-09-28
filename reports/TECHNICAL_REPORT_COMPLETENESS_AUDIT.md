# Technical Report Completeness Audit

## Measured deliverable statistics

- Rendered PDF page count: 80 (visually inspected after DOCX-to-PDF render)
- Canonical Markdown word count: 15554
- Chapters: 27
- Main-text figures: 34
- Appendix figures: 17
- Total unique embedded figures: 51
- Tables indexed: 11
- Equations explicitly explained: 7
- Algorithms/pseudocode blocks: 0 (implementation is described as prose; no unsupported pseudocode is presented as executed code)

## Evidence and quality checks

- Evidence manifest: `technical_report_evidence_manifest.csv`
- Figure/table indexes: generated from actual repository files
- Segmentation claims: limited to method because local results are `not_trained`
- Regression metrics: traced to raw saved predictions and recomputation
- Classification: corrected from hard labels; no probability-only metrics claimed
- Statistical analysis: 42 shared IDs, image-level bootstrap, paired Wilcoxon, Holm correction

## Chapter-length qualification

Individual chapter lengths vary because the report avoids fabricating missing segmentation output, model-architecture details and unavailable experimental metadata. The appendices contain the available scientific figures with source-bound interpretations so the report remains self-contained without treating figure count as proof of experimental scope.
