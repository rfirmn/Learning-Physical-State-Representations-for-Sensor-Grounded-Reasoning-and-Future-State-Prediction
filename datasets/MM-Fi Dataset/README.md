# MM-Fi filtered mmWave dataset

`filtered_mmwave` adalah sumber frame radar yang dipakai oleh loader penelitian ini. Format yang didukung adalah `.bin` dan satu `ground_truth.npy` per rekaman aksi.

## Struktur direktori

```text
filtered_mmwave/
├── E01/ ... E04/                 # lingkungan
│   └── S01/ ... S40/             # subjek
│       └── A01/ ... A27/         # aksi
│           ├── frame001.bin      # frame radar float32
│           ├── frame002.bin
│           └── ground_truth.npy  # array (F, 17, 3)
└── ...
```

`MMFi_action_segments.csv` berada satu tingkat di atas folder `filtered_mmwave` dan menyimpan rentang segmen repetisi yang digunakan saat membentuk jendela temporal. Rentang yang terbalik atau tidak konsisten harus dikarantina oleh pipeline; jangan dibetulkan diam-diam.

## Format frame dan ground truth

Setiap file `frame*.bin` dibaca sebagai array `float32` dan diubah menjadi matriks `(N, 5)` dengan kolom:

```text
x, y, z, doppler, snr
```

Loader memfilter nilai nonfinite/sentinel, kemudian melakukan sampling atau padding ke `N=128` titik. Ekstraktor fitur recovery mengunci sampling dengan seed dan menyimpan seed tersebut dalam provenance. Jika `use_extra_features: false`, hanya `x, y, z` yang diteruskan ke encoder; konfigurasi penelitian menggunakan kelima kanal.

`ground_truth.npy` harus berbentuk `(F, 17, 3)` dan frame ke-`k` pada nama `frame{k}.bin` dipasangkan dengan indeks `k-1` pada array ground truth. Ekstraksi fitur recovery menolak GT yang hilang, kosong, nonfinite, atau tidak mencakup frame yang diminta.

## Split penelitian

Konfigurasi cross-subject yang digunakan oleh Tahap 2–4 adalah:

- Train: `S01 S02 S03 S06 S08 S09 S11 S12 S14 S15 S16 S19 S21 S23 S26 S27 S29 S30 S31 S32 S33 S37 S38 S39`
- Validation: `S05 S10 S18 S20 S24 S28 S34 S35`
- Held-out test: `S04 S07 S13 S17 S22 S25 S36 S40`

Split ini menahan subjek test dari training. Ia tidak dengan sendirinya membuktikan robustness terhadap lingkungan baru karena semua split mencakup lingkungan `E01`–`E04`.

## Penggunaan pipeline

Loader utama berada di `eksperimen_model/datasets/mmfi_dataset.py`. Untuk mengekstrak latent `Z_t` pada recovery Stage 4, jalankan dari root repositori:

```powershell
& ".venv\Scripts\python.exe" eksperimen_model\datasets\extract_physical_features.py `
    --config eksperimen_model\configs\mmfi_dynamics_v3.yaml `
    --output_dir datasets\MM-Fi_features_stage4_recovery `
    --seed 42
```

Ekstraksi membutuhkan checkpoint encoder Av2, file segmentasi, dan seluruh frame/GT yang dirujuk konfigurasi. Artefak fitur menyimpan ID frame sumber serta provenance encoder, preprocessing, seed sampling, dan manifest data agar cache yang tidak cocok ditolak.

Dataset asli MM-Fi memiliki lisensi dan aturan sitasi sendiri. Periksa sumber resmi dataset sebelum mendistribusikan ulang data atau subsetnya.
