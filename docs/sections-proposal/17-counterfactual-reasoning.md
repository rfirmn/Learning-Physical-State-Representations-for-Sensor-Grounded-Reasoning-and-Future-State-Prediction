17. Counterfactual reasoning

Counterfactual evaluation menjadi salah satu evaluasi grounding terpenting.

Contoh:

Original

A:
position = (1,2)
velocity = (+1,0)
B:
position = (4,2)
velocity = (0,0)

Model memprediksi:

A approaches B

Kemudian physical state diubah:

Counterfactual

A:
position = (1,2)
velocity = (-1,0)

Model seharusnya menghasilkan:

A moves away from B

Untuk future prediction:

Original
     ↓
Trajectory A₁
Counterfactual
     ↓
Trajectory A₂

Trajectory harus berubah secara konsisten dengan intervention.
