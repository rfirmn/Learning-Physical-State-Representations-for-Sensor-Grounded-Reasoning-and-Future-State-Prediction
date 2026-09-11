22. Ablation study

Komponen dapat dihilangkan satu per satu:

Full Model
    │
    ├── remove Doppler
    ├── remove temporal information
    ├── remove velocity
    ├── remove dynamics model
    ├── remove alignment
    ├── remove LoRA
    └── reduce temporal context

Tujuannya bukan sekadar mendapatkan score tertinggi.

Tujuannya adalah mengetahui:

Informasi fisik apa yang sebenarnya diperlukan untuk reasoning dan prediction?

Contoh hasil hipotetis:

Full model       82%
-Doppler         74%
-Temporal        66%
-Dynamics        71%
-LoRA            80%

Interpretasinya dapat menjadi:

Temporal information dan Doppler memberikan kontribusi yang lebih besar terhadap physical reasoning dibandingkan adaptation pada LLM.

Itulah insight penelitian.
