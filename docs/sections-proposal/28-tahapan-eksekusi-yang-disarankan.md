28. Tahapan eksekusi yang disarankan

Phase 1 — Feasibility

Dataset
 ↓
Preprocessing
 ↓
Simple encoder
 ↓
Physical state estimation

Target:

memastikan radar dapat digunakan untuk memprediksi physical state.

Jika tahap ini gagal, jangan lanjut ke LLM.

--------

Phase 2 — Dynamics

Sensor sequence
 ↓
Encoder
 ↓
Latent state
 ↓
Dynamics model
 ↓
Future state

Target:

membuktikan bahwa representation memiliki predictive information.

--------

Phase 3 — LLM

Physical representation
 ↓
Projection
 ↓
LLM
 ↓
Physical reasoning

Target:

membuktikan LLM dapat menggunakan physical representation.

--------

Phase 4 — Grounding

Original
vs
Counterfactual

Target:

memastikan reasoning benar-benar bergantung pada physical information.

--------

Phase 5 — Robustness

In-domain
vs
Out-of-domain

Target:

menguji apakah representation menangkap physical structure atau hanya menghafal pattern.
