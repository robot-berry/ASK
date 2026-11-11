# Final objective quality gate

- Evaluation: `D:\robot\Documents\ASK\picture\test1\output\evaluations\agentpicture_cleanup_release`
- Manifest: `results.csv`
- Samples: 20; successful: 20
- Cleanup methods: {'matched-electronic': 20}
- Reliable electronic-reference pairs: 20

## Reference-backed metrics

- Mean absolute direction residual: 0.00 deg
- P95 absolute direction residual: 0.00 deg
- Residual >2 deg: 0
- Mean content retention: 100.00%
- Minimum content retention: 100.00%
- Mean reference content inside output: 100.00%
- Minimum reference content inside output: 100.00%
- Mean background false ink: 0.00%
- Maximum background false ink: 0.00%

## Photographic/self metrics

- Photographic samples: 0
- Reliable self-direction estimates: 0
- Mean absolute self direction: N/A
- Mean small noise components/Mpix: N/A
- Maximum small noise components/Mpix: 0.00

## Question-card coverage

- Mean output-ink coverage: 100.00%
- Minimum output-ink coverage: 100.00%
- Samples without cards: []

## Gate

Status: **PASS**; complete evidence: False

| Check | Observed | Requirement | Result |
|---|---:|---:|---:|
|success_rate|1.0|>= 1.0|PASS|
|reference_mean_abs_direction_deg|0.0|<= 0.8|PASS|
|reference_over_2deg_count|0|<= 1|PASS|
|reference_mean_background_false_ink_rate|0.0|<= 0.015|PASS|
|reference_mean_content_retention|1.0|>= 0.98|PASS|
|minimum_question_card_ink_coverage|1.0|>= 0.95|PASS|
|minimum_reference_content_coverage|1.0|>= 0.95|PASS|
|maximum_self_small_noise_components_per_mpix|0.0|<= 8000.0|N/A|
