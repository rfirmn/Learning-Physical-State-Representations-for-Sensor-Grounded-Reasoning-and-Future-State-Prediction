16. Task reasoning

Setelah representasi fisik berhasil dipelajari dan disejajarkan melalui modul proyektor, model bahasa (Frozen SLM) digunakan untuk melakukan *sensor-grounded physical reasoning* terhadap subjek yang diamati radar:

### Task 1 — State & Posture Reasoning
* Bagaimana konfigurasi postur tubuh subjek saat ini (berdiri, duduk, membungkuk)?
* Di mana posisi relatif lengan terhadap torso dan kepala?
* Apakah kedua pergelangan tangan berada di atas atau di bawah bahu?

### Task 2 — Spatial Geometry Reasoning (Body-Relative Geometry)
* Berapa rasio jarak antar-pergelangan tangan relatif terhadap lebar bahu (*current wrist separation*: narrow, medium, wide)?
* Apakah lengan kiri berada lebih tinggi atau lebih rendah daripada lengan kanan?
* Apakah orientasi tubuh condong ke arah kiri atau kanan relatif terhadap posisi radar?

### Task 3 — Temporal & Kinematic Reasoning
* Apakah anggota tubuh sedang bergerak mendekat (*positive Doppler*) atau menjauh (*negative Doppler*) terhadap sensor?
* Apakah gerakan tangan mengalami percepatan atau deselerasi antara frame $t_1$ dan $t_2$?
* Berapa fase siklus pergerakan yang telah dilalui subjek selama rentang observasi 1.6 detik?

### Task 4 — Relational Kinematics Reasoning
* Apakah kedua tangan bergerak secara simetris atau asimetris?
* Apakah jarak antara tangan dan torso sedang merenggang atau menyempit?
* Bagaimana koordinasi pergerakan antara tungkai bawah dan lengan atas?

### Task 5 — Future-State Reasoning
* Bagaimana perubahan separasi pergelangan tangan dalam rentang 0.8 detik ke depan (*future wrist separation change*: narrowing, unchanged, widening)?
* Apakah subjek diprediksi akan menyelesaikan gerakan mengangkat tangan atau kembali ke posisi istirahat?
* Apakah lintasan dinamika masa depan konsisten dengan momentum dan inersia yang diobservasi pada jendela historis?
