# Phase 1.6 — Prospective replication on subjects 03–05

This is a prospective replication of mean-model row-rank disagreement discovered using subjects 01–02, not an original preregistration. All secondary metrics/models and budgets are retained. No new scoring definition, preprocessing choice, checkpoint or budget was introduced.

Protocol snapshot Git commit: `ac462a65b49158cf63311c6c2dfb596f7cc86af2`; parent checkout commit: `51793f971498dcb03271d5a9f139fa1ce5302424`. Snapshot SHA256: `20840d3661ce9c75e39c1290887b8e8f0b83eebe18150d93b79c268c6654be6e`. Exact source/config hashes and timestamp are in `frozen_protocol.json`; the immutable Git object contains `protocol_snapshot.tar`. The branch and checkout were not replaced. This snapshot preceded new-subject network access and EEG inspection.

## Acquisition, schema and reliability

Only the official standard subject-03/04/05 archives were requested, with fresh bounded official OSF metadata checks against independently recorded trusted hash/size pins and full downloaded ZIP SHA256 verification. Standard ZIPs contain both test and training members; only the test member was extracted or processed. No separate training EEG or 63-channel archive was acquired. Verified archives, extracted object arrays and trusted metadata are preserved. Previously verified images and image metadata are reused.

Conversion uses the unchanged approved UID-1000, descriptor-only, default-deny seccomp child, with no filesystem opens, networking or process creation after confinement; all three denial probes run before pickle loading. Resource limits: 4 GiB address space, CPU 120 seconds, file size 512 MiB, parent timeout 150 seconds. All subsequent loads use allow_pickle=False.

| Subject | Cohort | Status | Brain split-half median | 95% split interval | Reliability p | Within primary score-rank median |
|---|---|---|---:|---|---:|---:|
| sub01 | discovery | verified | 0.7065 | [0.6775, 0.7236] | 0.000999 | 0.8548 |
| sub02 | discovery | verified | 0.6466 | [0.6251, 0.6694] | 0.000999 | 0.8021 |
| sub03 | replication | verified | 0.8145 | [0.7974, 0.8268] | 0.000999 | 0.8387 |
| sub04 | replication | verified | 0.7663 | [0.7477, 0.7814] | 0.000999 | 0.8694 |
| sub05 | replication | verified | 0.7335 | [0.7172, 0.7489] | 0.000999 | 0.8335 |

Successful new subjects have actual float64 EEG (200,80,17,100), 80 repetitions, identical 17 channel names/order and 100-Hz times −0.20 to 0.79 s. Every numeric condition maps to the exact existing image path; all 200 ordered IDs match discovery. Each subject has a shape report, mapping audit, 680-dimensional features, Brain RDM and split-half outputs. Failures are retained, never replaced or silently excluded; incomplete cohorts cannot receive a prospective GO decision.

Brain features use the unchanged [100,500) ms window and all 17 supplied posterior channels, no added baseline, bins or standardization. Brain RDM correlation, model RDM cosine and pinned DINOv2/CLIP embeddings are identical to discovery. Thirty 40/40 repetition splits use seed 20261011; split intervals describe variability, not independent sessions or population confidence intervals.

## Pairwise Brain RDM and disagreement results

| Pair | Group | Brain RDM ρ | Primary score ρ | Shared-brain null median | Shared p | Partial rank correlation | Partial shared p |
|---|---|---:|---:|---:|---:|---:|---:|
| sub01–sub02 | discovery_pair | 0.5495 | 0.7435 | 0.5113 | 0.001998 | 0.6927 | 0.007992 |
| sub01–sub03 | discovery_new | 0.5931 | 0.7817 | 0.5298 | 0.000999 | 0.7532 | 0.000999 |
| sub01–sub04 | discovery_new | 0.5048 | 0.6611 | 0.4429 | 0.004995 | 0.5865 | 0.018981 |
| sub01–sub05 | discovery_new | 0.5596 | 0.6813 | 0.5162 | 0.011988 | 0.6353 | 0.048951 |
| sub02–sub03 | discovery_new | 0.4612 | 0.6644 | 0.4410 | 0.004995 | 0.6093 | 0.013986 |
| sub02–sub04 | discovery_new | 0.4189 | 0.6126 | 0.3896 | 0.001998 | 0.5069 | 0.065934 |
| sub02–sub05 | discovery_new | 0.5740 | 0.6837 | 0.5418 | 0.028971 | 0.6217 | 0.150849 |
| sub03–sub04 | new_new | 0.5676 | 0.7010 | 0.5083 | 0.002997 | 0.6509 | 0.013986 |
| sub03–sub05 | new_new | 0.6010 | 0.6508 | 0.5530 | 0.111888 | 0.6005 | 0.267732 |
| sub04–sub05 | new_new | 0.4551 | 0.5849 | 0.4133 | 0.017982 | 0.4850 | 0.155844 |

Pairwise correlations share subjects and images and are dependent. No pair-based t-test, standard error or population confidence interval is calculated. Discovery pair 01–02, the three new-new pairs, discovery-new pairs, and all-five descriptive results are distinguished.

- discovery_pair: 1 dependent pairs; descriptive Brain median 0.5495, primary ranking median 0.7435, partial median 0.6927.
- new_new: 3 dependent pairs; descriptive Brain median 0.5676, primary ranking median 0.6508, partial median 0.6005.
- discovery_new: 6 dependent pairs; descriptive Brain median 0.5322, primary ranking median 0.6728, partial median 0.6155.
- All available five-subject pairs: descriptive Brain median 0.5546, primary rank median 0.6728; this full-pool description includes discovery and is not a new-subject confirmatory test.

## Frozen brain-specific and residual controls

Independent-brain controls draw one independent assignment for each of all five subjects per permutation; each Brain RDM is relabeled on BOTH axes and every score is recomputed against fixed model RDMs. All five assignments are generated even if a subject fails. Shared-brain controls use one identical assignment for every subject, preserving between-brain geometry while disrupting model correspondence. These are not score-label shuffles. The shared-null excess is not a causal decomposition.

Seeds/counts: 1,000 joint independent draws at 2026101501; 1,000 shared permutations at 2026101502; 1,000 tie-aware score-label permutations at 2026101503. The shared draws match Phase 1.5; independent draws extend the same seed to coherent five-subject assignments, so discovery independent Monte Carlo samples differ from the earlier two-subject run. Definitions and counts remain frozen.

Model-only covariates and residualization are unchanged: DINOv2–CLIP row disagreement; each model nearest-20 mean cosine distance; each model all-nonself mean cosine distance; ranked/standardized controls plus intercept. Partial rank association is Pearson correlation of residualized score ranks, not a predictive model. The fixed model-control projection is reapplied to every null score. In-sample explained rank variance is descriptive only.
- New-new joint cohort primary_independent: observed median 0.6508, null median -0.0055, 95% null interval [-0.1166, 0.1113], p=0.000999.
- New-new joint cohort primary_shared: observed median 0.6508, null median 0.4944, 95% null interval [0.3589, 0.6237], p=0.011988.
- New-new joint cohort partial_independent: observed median 0.6005, null median -0.0028, 95% null interval [-0.1105, 0.0989], p=0.000999.
- New-new joint cohort partial_shared: observed median 0.6005, null median 0.4953, 95% null interval [0.3590, 0.6204], p=0.047952.
- Prospective primary Holm-adjusted p-values (full ranking, partial ranking): [0.023976023976023976, 0.04795204795204795]. The null statistic takes the median of the three dependent new-new pairs for EACH joint permutation; it does not treat pairs as independent replicates.

## Selection-budget replication

All five budgets and every method/model/subject are reported in selection_budget_replication.csv. Within-subject medians and split quantiles use disjoint repetition halves. Cross-subject full-score values have separate half-RDM variability summaries; those are not confidence intervals for full 80-repetition scores. Nulls reapply the exact fixed-ID tie rule after relabeling/recomputation. Negative median-null-adjusted overlap is retained. This adjustment is (observed−median null)/(1−median null), not mean-chance kappa.

| Budget | New-new median observed overlap | Shared joint-null median | Shared joint-null 95% interval | Shared p | Holm p across five budgets |
|---|---:|---:|---|---:|---:|
| 10 | 0.3000 | 0.2000 | [0.1000, 0.5000] | 0.462537 | 0.569431 |
| 20 | 0.5000 | 0.3500 | [0.2000, 0.5000] | 0.038961 | 0.194805 |
| 40 | 0.5000 | 0.4500 | [0.3250, 0.5750] | 0.248751 | 0.569431 |
| 60 | 0.5833 | 0.5333 | [0.4333, 0.6333] | 0.189810 | 0.569431 |
| 100 | 0.7300 | 0.6800 | [0.6100, 0.7400] | 0.062937 | 0.251748 |

No optimal budget is selected. High whole-ranking correlation does not imply a stable narrow shortlist. Inspect every new-subject pair, model variant, split interval and independent/shared null in the CSV/JSON, including unfavorable findings. Cohort budget statistics use a median per joint draw, preserving dependence among pairs.

## Frozen decisions

- **A. Extend to all ten subjects: GO.**
- **B. Investigate reliability-aware scoring separately: GO, separate exploratory investigation only.**
- **C. Design an independent-pool selection experiment: NO-GO.**

The decision rules were frozen before acquisition: A requires all three subjects verified, reliable Brain RDMs, all new-new primary correlations positive, and both primary Holm p≤0.05. B requires all three verified reliable datasets and authorizes only a separate exploratory investigation. C requires A and at least one Holm-adjusted cohort budget excess; it authorizes study design only, with an independent stimulus pool and independently fitted preprocessing. No new reliability-aware score was implemented, no next subject was downloaded, and no training or data-efficiency experiment was executed.

## Limitations and reproducibility

All EEG findings are conditional on the official shared MVNN whitening, which combined training/test partitions and repetitions during covariance fitting. Random disjoint halves are not independently preprocessed or session-independent. Raw event roundtrip/session labels and training-pixel alias audits are unavailable. Image-label permutations can disrupt image-specific confounds as well as genuine alignment; five model covariates do not exhaust visual/semantic confounds. Discovery-informed candidate selection is explicitly acknowledged; prospective replication does not remove preprocessing limitations.

Only the two prospective cohort primary tests and five cohort budget tests receive their predeclared Holm corrections. All other method/model/pair p-values are secondary, exploratory and unadjusted with Monte Carlo resolution 1/1001. Population generalization, causality, classifier usefulness and improved data efficiency are not established. The original 200-image test split must remain protected from future training decisions.

Exact full/half scores, joint assignments and raw null statistics are retained in pickle-free NPZs. frozen_protocol.json pins Git snapshot, exact sources/config, dependency versions and failure policy. Per-subject acquisition manifests pin official metadata, full ZIP/test-member hashes and confinement/export evidence. analysis_provenance.json and verified_outputs.json provide input/output checksums. See VALIDATION.md and test_results.json for executed commands and tests.

## Interpretation of unfavorable and borderline replication findings

The new-new median full ranking association is 0.6508, lower than discovery's
0.7435. The prospective cohort test passes its frozen Holm rule (p=0.0240), but
this is **cohort-level support, not uniformly successful pairwise replication**.
Subject-03/05 full ranking does not clearly exceed its shared-brain null
(unadjusted p=0.1119). Only subject-03/04 has a clear unadjusted partial shared-null
excess (p=0.0140); subject-03/05 and subject-04/05 partial comparisons have
p=0.2677 and p=0.1558. All these unfavorable secondary results are retained.

The cohort partial association (0.6005) has Holm p=0.0480, barely below the frozen
0.05 threshold. Forty-seven of 1,000 shared-null draws exceed it. A descriptive
exact binomial 95% interval for the underlying sampled permutation-tail probability
is [0.0347, 0.0620], which spans 0.05. This quantifies Monte Carlo sampling
uncertainty, not effect/population uncertainty, and is not an extra primary test.
No permutations were added and no threshold or decision rule was changed. The
predeclared A decision is GO, with borderline incremental evidence explicitly
acknowledged; stronger confirmation is still needed.

The discovery budget-100 shortlist result does **not** clearly replicate at the
prospective cohort level: new-new median overlap is 73/100, shared-null median
68/100, raw p=0.0629 and Holm p=0.2517. Budget 20 has raw p=0.0390 but Holm
p=0.1948. None of the five budgets passes the frozen correction. Global ranking
agreement and reliable EEG therefore do not validate a selection policy.

A. GO for finishing the ten-subject reliability/ranking replication with the same
protocol and transparent unfavorable-subject reporting; this recommendation does
not authorize any further download in the current task.

B. GO for a separately labeled exploratory investigation of reliability-aware
scoring. Keep this prospective analysis unchanged and do not promise that weighting
will resolve subject differences or improve selection.

C. NO-GO at this stage for advancing the independent-pool selection experiment
under the frozen evidence rule. First establish reproducible selections beyond
shared-brain controls and plan independently fitted/session-separated preprocessing.
No budget or scoring winner is selected, and no model training or data-efficiency
claim follows from these results.
