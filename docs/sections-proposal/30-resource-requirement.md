30. Resource requirement

Tidak membutuhkan:

* training LLM from scratch;
* GPU cluster;
* massive dataset;
* robot;
* physical hardware deployment.

Dapat menggunakan:

Public dataset
+
Consumer GPU / Cloud GPU
+
Small sensor encoder
+
Small dynamics model
+
Frozen open-weight LLM
+
Optional LoRA

Komputasi terbesar diharapkan berada pada training encoder/dynamics, bukan full LLM training.
