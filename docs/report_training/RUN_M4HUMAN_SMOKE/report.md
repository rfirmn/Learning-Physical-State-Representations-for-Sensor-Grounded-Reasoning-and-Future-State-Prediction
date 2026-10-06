# M4Human E: RUN_M4HUMAN_SMOKE

Status: complete. Data: m4human.

Real-data gates and scientific claims require separate validation.

```json
{
  "contract_version": "m4human_kinetok_v3",
  "stage": "E",
  "run_id": "RUN_M4HUMAN_SMOKE",
  "data_kind": "m4human",
  "status": "complete",
  "elapsed_s": 1211.7649999999994,
  "resources": {
    "process_peak_rss_bytes": 7088918528,
    "cuda_peak_allocated_bytes": 95119872,
    "cuda_peak_reserved_bytes": 130023424
  },
  "validation": {
    "relative": {
      "error_sum": 140749.41990613937,
      "count": 1423779,
      "mean_m": 0.09885622691874187
    },
    "global": {
      "error_sum": 187812.71703863144,
      "count": 1491578,
      "mean_m": 0.12591545131306003
    },
    "root": {
      "error_sum": 5670.467649325728,
      "count": 67799,
      "mean_m": 0.08363644964270459
    },
    "global_per_joint": {
      "error_sum": [
        5670.467651735991,
        6029.764413610101,
        6026.030529718846,
        6061.556443508714,
        8483.769045390189,
        8302.906052496284,
        6306.049942348152,
        9727.775435492396,
        9798.550449594855,
        6389.226078443229,
        10805.470043756068,
        10692.384156074375,
        6292.13987782225,
        6068.016937684268,
        6158.891106631607,
        7041.028648234904,
        7770.677630148828,
        7601.875779129565,
        10866.426930654794,
        11631.736029490829,
        14604.810584135354,
        15483.163299761713
      ],
      "count": [
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799,
        67799
      ]
    }
  },
  "successful_updates": 28646,
  "selected_epoch": 0,
  "root_gate": {
    "threshold_m": null,
    "passed": null
  },
  "scientific_freeze_eligible": false,
  "checkpoint_sha256": "6942fc8350f6a2080ef93717348fdbc22d8e106e9d313c4a75e3e5797a3710ed",
  "test_status": "not_opened",
  "loss_curve_created": true,
  "analysis_ready": true,
  "artifact_issues": []
}
```
