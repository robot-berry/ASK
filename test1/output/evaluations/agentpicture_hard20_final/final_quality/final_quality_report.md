# Final objective quality gate

- Evaluation: `D:\robot\Documents\ASK\picture\test1\output\evaluations\agentpicture_hard20_final`
- Manifest: `results.csv`
- Samples: 20; successful: 20
- Cleanup methods: {'unknown': 20}
- Reliable electronic-reference pairs: 18

## Reference-backed metrics

- Mean absolute direction residual: 1.54 deg
- P95 absolute direction residual: 5.78 deg
- Residual >2 deg: 4
- Mean content retention: 85.87%
- Minimum content retention: 36.16%
- Mean reference content inside output: 99.17%
- Minimum reference content inside output: 85.02%
- Mean background false ink: 3.42%
- Maximum background false ink: 11.16%

## Photographic/self metrics

- Photographic samples: 0
- Reliable self-direction estimates: 0
- Mean absolute self direction: N/A
- Mean small noise components/Mpix: N/A
- Maximum small noise components/Mpix: 0.00

## Question-card coverage

- Mean output-ink coverage: 67.87%
- Minimum output-ink coverage: 22.07%
- Samples without cards: []

## Gate

Status: **FAIL**; complete evidence: False

| Check | Observed | Requirement | Result |
|---|---:|---:|---:|
|success_rate|1.0|>= 1.0|PASS|
|reference_mean_abs_direction_deg|1.54424|<= 0.8|FAIL|
|reference_over_2deg_count|4|<= 1|FAIL|
|reference_mean_background_false_ink_rate|0.034209|<= 0.015|FAIL|
|reference_mean_content_retention|0.858663|>= 0.98|FAIL|
|minimum_question_card_ink_coverage|0.220742|>= 0.95|FAIL|
|minimum_reference_content_coverage|0.850217|>= 0.95|FAIL|
|maximum_self_small_noise_components_per_mpix|0.0|<= 8000.0|N/A|
