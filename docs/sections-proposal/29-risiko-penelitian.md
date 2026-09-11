29. Risiko penelitian

Risiko 1 — Physical state prediction buruk

Solusi:

* sederhanakan target;
* gunakan subset dataset;
* mulai dari position/velocity;
* jangan langsung predict full skeleton.

--------

Risiko 2 — Dynamics prediction terlalu sulit

Solusi:

gunakan horizon pendek terlebih dahulu:

t → t+1

kemudian:

t → t+1...t+k

--------

Risiko 3 — LLM tidak dapat menggunakan representation

Solusi:

gunakan structured physical representation sebagai intermediate baseline.

Radar
 ↓
Physical State
 ↓
Text / Structured Representation
 ↓
LLM

Jika pipeline ini berhasil, baru bandingkan dengan learned representation.

--------

Risiko 4 — LLM terlalu dominan

Solusi:

gunakan frozen LLM sebagai baseline dan pisahkan kontribusi sensor-side dengan LLM-side.

--------

Risiko 5 — Dataset terlalu sederhana

Solusi:

mulai dari dataset controlled untuk feasibility, kemudian gunakan M4Human sebagai extension.
