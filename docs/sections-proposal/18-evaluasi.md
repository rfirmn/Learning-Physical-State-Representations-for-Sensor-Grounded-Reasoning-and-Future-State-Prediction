18. Evaluasi

Penelitian tidak hanya menggunakan accuracy.

A. Physical State Error

Untuk posisi atau koordinat:

[
MAE = \frac{1}{N}\sum |y-\hat{y}|
]

-------

B. Velocity Error

Mengukur error prediksi velocity:

[
MAE_v = \frac{1}{N}\sum |v-\hat{v}|
]

-------

C. Direction Accuracy

Membandingkan:

Predicted direction
vs
Ground-truth direction

-------

D. ADE

Untuk trajectory prediction:

[
ADE = \frac{1}{T}\sum_{t=1}^{T}
||p_t-\hat p_t||
]

Mengukur rata-rata error trajectory.

--------

E. FDE

[
FDE = ||p_T-\hat p_T||
]

Mengukur seberapa jauh prediction terakhir dari ground truth.

---------

F. Spatial / Temporal Reasoning Accuracy

Mengukur kemampuan menjawab:

left/right
near/far
approaching/away
increasing/decreasing

--------

G. Counterfactual Consistency

Mengukur apakah perubahan physical state menghasilkan perubahan prediction/reasoning yang benar.

[
CF\ Consistency =
\frac{
Correctly\ Changed\ Predictions
}{
Total\ Counterfactual\ Cases
}
]
