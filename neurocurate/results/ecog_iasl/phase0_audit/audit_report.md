# DS004194 ECoG metadata audit

Status: **E0_PASS**

- NEMAR version: v1.0.0; OpenNeuro source: ds004194 v3.0.0
- License: CC0
- Metadata records: 209; canonical VHDR recordings: 209
- Dataset payload size: 57.9 GB (not downloaded)
- Downloaded during audit: only metadata, events.tsv and channels.tsv; no iEEG payload.

## Subject/task coverage

- `prf`: 13 subjects (p01, p02, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14)
- `sixcatlocdiffisi`: 2 subjects (p13, p14)
- `sixcatlocisidiff`: 1 subjects (p12)
- `sixcatloctemporal`: 4 subjects (p11, p12, p13, p14)
- `soc`: 2 subjects (p03, p04)
- `spatialpattern`: 9 subjects (p01, p02, p05, p06, p07, p08, p09, p10, p11)
- `temporalpattern`: 9 subjects (p01, p02, p05, p06, p07, p08, p09, p10, p11)

## Common conditions

- `prf`: 13 subjects, 0 common trial names
- `sixcatlocdiffisi`: 2 subjects, 72 common trial names
- `sixcatlocisidiff`: 1 subjects, 72 common trial names
- `sixcatloctemporal`: 4 subjects, 72 common trial names
- `soc`: 2 subjects, 36 common trial names
- `spatialpattern`: 9 subjects, 12 common trial names
- `temporalpattern`: 9 subjects, 12 common trial names

## E0 decision

The audit does not treat all 14 participants as interchangeable. A common task is eligible only when at least four subjects share the task and event condition. Electrode indices are not directly comparable; channels.tsv provides local channel quality but no universal visual ROI label. Visual electrode masks require task-specific derivative coordinates or fold-local response selection.

No SAE, intervention or LOSO training was run before this audit.
