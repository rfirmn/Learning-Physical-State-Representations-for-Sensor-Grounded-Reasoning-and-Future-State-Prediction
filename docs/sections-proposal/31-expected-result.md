31. Expected result

Ada beberapa kemungkinan hasil yang semuanya valid.

Hasil A — Representation + dynamics berhasil

Sensor representation
        ↓
Good future prediction
        ↓
Better physical reasoning

Kesimpulan:

Learned representation mampu mempertahankan physical state dan dynamics yang berguna untuk reasoning.

--------

Hasil B — Representation bagus untuk state tetapi buruk untuk future

State estimation ↑
Prediction ↓

Ini menunjukkan:

representation menangkap physical state tetapi belum menangkap underlying dynamics secara memadai.

Ini merupakan hasil yang sangat relevan untuk world modeling.

--------

Hasil C — Dynamics bagus tetapi LLM tidak mampu menggunakannya

Physical prediction ↑
LLM reasoning ↓

Kesimpulan:

physical representation dapat memodelkan dynamics, tetapi sensor-language interface masih menjadi bottleneck.

--------

Hasil D — Counterfactual gagal

Jika:

Physical state changes
        ↓
Prediction barely changes

maka:

model mungkin mempelajari correlation daripada physical dynamics.

Ini juga merupakan hasil penelitian yang valid.

--------

Hasil E — Robustness rendah

Jika:

In-domain ↑
Out-of-domain ↓↓↓

maka:

representation belum cukup invariant terhadap environmental changes.

Ini membuka penelitian lanjutan mengenai robust physical representation.
