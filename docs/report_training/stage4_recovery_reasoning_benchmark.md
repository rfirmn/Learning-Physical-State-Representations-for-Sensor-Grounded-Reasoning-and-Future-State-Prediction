# Stage 4 matched reasoning benchmark

Test QA SHA-256: 53575657085128c3e60bc972c251e601de21d59d60039f3e1240b26f963a961b

| Condition | Task | N | Parsed | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| B1 | current_wrist_separation | 2949 | 0 | 0.000 | 0.000 |
| B1 | future_wrist_separation_change | 3140 | 0 | 0.000 | 0.000 |
| B0 | current_wrist_separation | 2949 | 2949 | 0.874 | 0.466 |
| B0 | future_wrist_separation_change | 3140 | 3140 | 0.348 | 0.172 |
| B2 | current_wrist_separation | 2949 | 2949 | 0.888 | 0.646 |
| B2 | future_wrist_separation_change | 3140 | 3140 | 0.489 | 0.489 |
| B3 | current_wrist_separation | 2949 | 2948 | 0.926 | 0.791 |
| B3 | future_wrist_separation_change | 3140 | 3140 | 0.488 | 0.486 |
| B3P | current_wrist_separation | 2949 | 2949 | 0.927 | 0.792 |
| B3P | future_wrist_separation_change | 3140 | 3140 | 0.472 | 0.471 |
| B4 | current_wrist_separation | 2949 | 2949 | 0.921 | 0.764 |
| B4 | future_wrist_separation_change | 3140 | 3140 | 0.461 | 0.457 |
| B4_cross_action_shuffle | current_wrist_separation | 2949 | 2949 | 0.498 | 0.333 |
| B4_cross_action_shuffle | future_wrist_separation_change | 3140 | 3140 | 0.261 | 0.260 |
| B4_within_action_shuffle | current_wrist_separation | 1293 | 1293 | 0.856 | 0.726 |
| B4_within_action_shuffle | future_wrist_separation_change | 3084 | 3084 | 0.362 | 0.357 |

Primary paired B4 minus B3P macro-F1: -0.014 (95% recording-cluster interval -0.035 to +0.005; N=3140, recordings=152).

Primary paired comparison: B4 minus B3P on future wrist-separation change.
Additional ablations: B4 minus B3, B3P minus B3, and optional B5 minus B4.
See JSON for recording-cluster intervals, subject and repetition-boundary strata, raw responses, and shuffle donor mappings.
B2 is a direct probe when supplied; B5 is an optional observed-future diagnostic.
