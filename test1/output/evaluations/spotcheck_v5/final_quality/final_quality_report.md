# Final objective quality gate

- Evaluation: `D:\robot\Documents\ASK\picture\test1\output\evaluations\spotcheck_v5`
- Manifest: `reconstructed-markdown`
- Samples: 2; successful: 2
- Cleanup methods: {'photographic': 2}
- Reliable electronic-reference pairs: 1

## Reference-backed metrics

- Mean absolute direction residual: 2.18 deg
- P95 absolute direction residual: 2.18 deg
- Residual >2 deg: 1
- Mean content retention: 38.37%
- Minimum content retention: 38.37%
- Mean reference content inside output: 98.18%
- Minimum reference content inside output: 98.18%
- Mean background false ink: 6.83%
- Maximum background false ink: 6.83%

## Photographic/self metrics

- Photographic samples: 2
- Reliable self-direction estimates: 2
- Mean absolute self direction: 0.70 deg
- Mean small noise components/Mpix: 84.76
- Maximum small noise components/Mpix: 133.55

## Question-card coverage

- Mean output-ink coverage: 87.99%
- Minimum output-ink coverage: 75.99%
- Samples without cards: []

## Gate

Status: **FAIL**; complete evidence: True

| Check | Observed | Requirement | Result |
|---|---:|---:|---:|
|success_rate|1.0|>= 1.0|PASS|
|reference_mean_abs_direction_deg|2.175381|<= 0.8|FAIL|
|reference_over_2deg_count|1|<= 1|PASS|
|reference_mean_background_false_ink_rate|0.06828|<= 0.015|FAIL|
|reference_mean_content_retention|0.383708|>= 0.98|FAIL|
|minimum_question_card_ink_coverage|0.759873|>= 0.95|FAIL|
|minimum_reference_content_coverage|0.981758|>= 0.95|PASS|
|maximum_self_small_noise_components_per_mpix|133.550891|<= 8000.0|PASS|
