12. Sensor encoder

Beberapa kandidat:

* Temporal Transformer;
* Point Transformer;
* PointNet/PointNet++;
* lightweight Transformer;
* 1D CNN untuk structured temporal representation.

Pemilihan model tidak menjadi kontribusi utama.

Tujuannya adalah mendapatkan representation:

[
z_t = f(X_{t-k:t})
]

di mana:

* (X) = sensor observation;
* (z_t) = learned physical representation.
