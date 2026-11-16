# Technical report — outline (deliverable: PDF)

Host expectations (competition discussion #728302): a succinct but thorough,
step-by-step description of every pipeline component; no micro-level decision
undocumented (split protocol, downsampling ratios, normalization parameters,
seeds, environment, ...).

1. Problem & data — MAGFiLO Kaggle subset; multi-annotator structure; known
   public-LB caveats (empty-mask artifact, public-GT overlap) and why model
   selection therefore runs on an internal protocol.
2. Evaluation protocol — image-score definition, two GT protocols, empty
   semantics; inter-annotator ceiling as calibration.
3. Data handling & integrity — test-isolation guard (pinned list + tests);
   external-data accounting + ablation (public negatives only).
4. Preprocessing — resolution, normalization: exact parameters.
5. Model — architecture, encoder, loss, schedule, seeds, environment.
6. Instancing & post-processing — connected components, fragmentation
   handling: exact rules and thresholds.
7. Results — per-fold CV, score distributions, fragmentation stats, ablations.
8. Reproduction — exact commands, hardware, runtime.
