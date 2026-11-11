# Final objective quality gate

- Evaluation: `D:\robot\Documents\ASK\picture\test1\output\evaluations\spotcheck_v4`
- Manifest: `results.json`
- Samples: 5; successful: 5
- Cleanup methods: {'matched-electronic': 3, 'photographic': 2}
- Reliable electronic-reference pairs: 4

## Reference-backed metrics

- Mean absolute direction residual: 0.39 deg
- P95 absolute direction residual: 1.31 deg
- Residual >2 deg: 0
- Mean content retention: 84.16%
- Minimum content retention: 36.65%
- Mean reference content inside output: 99.66%
- Minimum reference content inside output: 98.65%
- Mean background false ink: 1.80%
- Maximum background false ink: 7.19%

## Photographic/self metrics

- Photographic samples: 2
- Reliable self-direction estimates: 2
- Mean absolute self direction: 5.22 deg
- Mean small noise components/Mpix: 91.49
- Maximum small noise components/Mpix: 139.40

## Question-card coverage

- Mean output-ink coverage: 95.01%
- Minimum output-ink coverage: 77.61%
- Samples without cards: []

## Gate

Status: **FAIL**; complete evidence: True

| Check | Observed | Requirement | Result |
|---|---:|---:|---:|
|success_rate|1.0|>= 1.0|PASS|
|reference_mean_abs_direction_deg|0.386371|<= 0.8|PASS|
|reference_over_2deg_count|0|<= 1|PASS|
|reference_mean_background_false_ink_rate|0.017972|<= 0.015|FAIL|
|reference_mean_content_retention|0.841629|>= 0.98|FAIL|
|minimum_question_card_ink_coverage|0.776074|>= 0.95|FAIL|
|minimum_reference_content_coverage|0.986542|>= 0.95|PASS|
|maximum_self_small_noise_components_per_mpix|139.398432|<= 8000.0|PASS|
