# Audit implementasi M4Human v3

Tanggal: 5 Oktober 2026, Asia/Jakarta. Ruang lingkup: kode baru M4Human sesuai lima planning, bukan audit ulang seluruh pipeline MM-Fi. Acuan keputusan ilmiah: `docs/proposal_riset_terbaru.md` dan `docs/planning_m4human/01_encoder.md` sampai `05_evaluasi_testing.md`.

## Kesimpulan dan batas bukti

Implementasi menyediakan jalur development P0 lengkap: audit/target offline, E, causal M, full-H compression dua kondisi, fresh probes, QA, projector ke frozen Qwen dan evaluasi terstruktur. Kode diuji dengan fixture sintetis dan ditinjau lintas subagent. Bukti eksekusi akhir berada pada [laporan verifikasi](m4human_verification_20261005_final.json); jalankan ulang bila hash source berbeda.

**Lulus tes implementasi tidak sama dengan valid secara empiris pada M4Human.** Dataset, aset SMPL-X yang relevan, pretrained Qwen dan RTX 3060 tidak tersedia dalam pengujian ini. Validitas semantik target, akurasi, efek C_kin, kemampuan grounding serta kelayakan resource tetap `not_run`. Tidak ada angka performa riset yang dihasilkan atau diklaim.

Eksekusi final: **6/6 modul lulus** pada Python 3.12.13, CPU macOS; 41 hash source/config/dependency cocok dengan berkas akhir dan tidak berubah selama suite. `compileall` dan `git diff --check` juga lulus. Laporan menyimpan stdout, exit code, durasi dan versi library untuk setiap pemeriksaan.

## Pemetaan implementasi terhadap planning

| Kontrak | Implementasi / bukti |
|---|---|
| Sensor-only, immutable LMDB, split sebelum fitting | `m4human_dataset.py`, `audit_m4human.py`; reader read-only, serializer bounded, multiprocessing spawn, leakage dan gap checks |
| Target 22 joint/pelvis, meter, transform audited | `export_m4human_targets.py`; transform round-trip, direct-joint equivalence gate, trusted SMPL-X asset hashes |
| E context 4 × 512 × 4, scratch, feature 256 | `m4human_encoder.py`; masked pooling, permutation/sampling, gradients dan tiny-overfit sintetis |
| M causal, H 32 × 23 × 128; Z diagnostik | `m4human_motion.py`; prefix invariance terhadap perubahan masa depan, masked NaN checks |
| Full-H fidelity pada kedua kondisi | `m4human_tokenizer.py`, `m4human_readout.py`; K8/K16/K32, chronological bins, query chunk equivalence |
| Auxiliary baseline M versus proposed U | Gradient-route checks dan separate clipping; auxiliary baseline yang diperbesar tidak mengubah update compressor |
| Physical p/v dan masker empat kelompok | `m4human_kinematics.py`; actual-time trailing polynomial derivative, analytic fixtures, SciPy cross-check, warmup/gap, select-before-arithmetic |
| Probe independen M/U | `train_m4human_probes.py`; inisialisasi dan training policy matched, source/cache lineage terikat checkpoint |
| QA core tasks dan support konsisten | H→QA→exact-U join; file hash, IDs, frame/time support, excluded/unknown, same recipe |
| Projector-only, frozen LLM | `m4human_projector.py`; answer-only causal shift, padding/position, gradient input tetap mengalir; parameter LLM tidak berubah |
| Generated output dan matched alignment | Tiny Qwen2 dari Transformers 4.45.2; cache/recompute generation parity, shared initial/alpha, exposure/update checks |
| Physical retention dan language scoring | Evaluator fisik, strict JSON parser, macro-F1 empat kelas, invalid-as-wrong, majority/text/rule/probe, donor coverage |
| Test tidak boleh menyesuaikan protokol | Lock checkpoint/cache/config/recipe/code dependency, effect tolerance, paired subject bootstrap policy |
| Resume dan rekonsiliasi laporan | Optimizer/RNG/scaler/sampler, interrupted versus uninterrupted bitwise checks, metric witness dari raw records |

## Temuan yang diperbaiki selama development

| Masalah | Perbaikan dan regresi |
|---|---|
| Hash normalizer tertimpa saat pindah tahap | Pisahkan upstream normalizer dan `stage_normalizer_hash`; full M→C→probe integration |
| Inferensi bergantung keberadaan label offline | Loader H inference tidak membaca target; ekstraksi E tidak membuka target/params; tes menyembunyikan target saat ekstraksi/loading |
| Masked NaN masuk LayerNorm sebelum sanitasi | Sanitasi cross-memory/query sebelum normalisasi; backward gradient harus finite |
| Auxiliary baseline dapat memengaruhi compressor melalui clipping global | Kelompok clipping terpisah; tes auxiliary scale besar dengan update compressor tetap sama |
| Fixed `position_ids` mengganggu native cached Qwen generation | Native generate menurunkan posisi dari attention mask; raw-first-logit, cached/recompute dan padded-batch checks |
| Identitas subjek integer dan hash joint-map tidak konsisten di QA | Registry menerima identitas dataset; generator memakai SHA-256 file joint-map seperti audit/E, dengan tes CLI H→QA→U |
| Cache/checkpoint asing dapat terlihat cocok hanya dari nama kondisi | Source/token cache, stats dan checkpoint integrity ikut diverifikasi; lock memeriksa run/provenance/cohort |
| Pilihan bootstrap CLI dapat melampaui protokol test | Draws/seed/method berasal dari locked contract pada test; undefined resamples tidak menghasilkan CI |
| Completion marker dapat menutupi nilai ringkasan yang berubah | Validator merekonsiliasi raw prediction → witness → metrics dan hash history/completion; tampered mean ditolak |

Audit independen dilakukan oleh subagent pada jalur data, motion dan evaluasi; parent mengintegrasikan dan menguji sambungan antarmodul. Temuan diperbaiki berdasarkan reproduksi/kontrak. Ini review internal development, belum peer review atau reproduksi oleh peneliti independen.

## Paket pemeriksaan yang dapat dijalankan ulang

```sh
.venv/bin/python -m eksperimen_model.test_m4human_pipeline --output PATH_BARU.json
.venv/bin/python -m eksperimen_model.validate_m4human_run RUN_DIRECTORY
```

Suite menjalankan enam modul contract/integration dengan Python yang sama. Artefak sementara tidak masuk daftar eksperimen ilmiah. File laporan mencatat stdout dan exit code setiap pemeriksaan, environment serta hash source/config. Pemeriksaan encoder mencakup audit 120 frame sintetis, target export dan ekstraksi state. Pemeriksaan motion menjalankan training M, dua C dan tiga fresh probes secara nyata dengan dataset kecil; language test memakai model Qwen2 kecil acak, bukan stub autograd semata. Jalur model/tokenizer pretrained tetap memerlukan pengujian terpisah.

Resume yang diuji memakai checkpoint pada batas epoch. Pengujian tidak membuktikan determinisme lintas perangkat, lintas versi PyTorch, kernel CUDA atau seluruh pola interruption. Karena itu run menyimpan environment dan config beserta state training, dan output resume memakai direktori baru.

## Gerbang ilmiah dan keterbatasan operasional

1. **Paket nyata belum diaudit.** Serializer publik adalah referensi; payload lokal harus cocok dengan schema aktual. Label unit, frame transform, regressor rows dan timestamp tidak dapat dibuktikan dengan shape check. Flag audit menandai bukti yang disediakan peneliti, bukan sensor kebenaran otomatis.
2. **SMPL-X nyata belum diuji.** Aset tidak diunduh. Jalur direct-joints ditahan oleh equivalence gate; fixture menguji mekanika, bukan ekuivalensi anatomi pada data nyata.
3. **Resource belum terukur pada RTX 3060.** Cache H/U memakai tensor fp32 exact-window; ukuran disk, RAM, throughput dan AMP parity harus diukur sebelum menetapkan budget. Tidak ada klaim 12 GB pasti cukup.
4. **Kualitas upstream menentukan interpretasi.** E/root, motion p/v dan independent-probe gates harus lulus train/val sebelum hasil downstream dianggap mendukung atau menolak hipotesis. Model gagal belajar bukan bukti negatif ilmiah.
5. **Inferensi utama memakai exact-U cache.** RPC→E→H→U tersedia sebagai tahap ekstraksi terpisah. Belum ada aplikasi streaming realtime atau service deployment; itu bukan syarat eksperimen P0.
6. **Klaim tambahan dibatasi.** K8/K32 didukung secara mekanis tetapi belum dibandingkan empiris. Onset/radial dan urutan kejadian tidak diaktifkan pada generator core. Tidak ada LoRA, forecasting, atau counterfactual fisik tersirat.
7. **Test lock bukan bukti belum pernah melihat hasil di luar sistem.** Ia mengikat artefak dan kebijakan eksekusi; disiplin penelitian tetap perlu mencatat kapan hasil test pertama dibuka. Bootstrap dilaporkan conditional pada checkpoint terlatih, dengan batas jumlah subjek/support; variasi seed belum diukur.

Untuk menjalankan gerbang tersebut pada komputer dataset, ikuti [runbook](m4human_development_runbook.md). Dokumen planning, perubahan proposal milik pengguna, checkpoint dan seluruh laporan historis tetap dipertahankan.
