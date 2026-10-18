# IASL Phase 0A data audit

- Commit: `92223e26ea97ad11b8eefdec6b44f823998b426e`
- Subjects: sub01, sub02, sub03, sub04, sub05
- Shared images: 50
- Repetitions used: 4
- Shape per subject: `(50, 4, 17, 80)` (images, repetitions, channels, time)
- Actual post-stimulus coverage: 0.00--0.79s; the requested 1.00s is unavailable in the verified export.
- All inputs are pickle-free numeric exports and all five subjects have the same 17-channel schema.

Repeated-trial reliability is reported in `data_manifest.json`; this is a feasibility audit, not a session-independent reliability claim.
