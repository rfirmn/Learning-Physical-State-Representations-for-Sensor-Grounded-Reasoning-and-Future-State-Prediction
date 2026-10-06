# Rencana perbaikan M4Human dan profiling RTX 3060 12 GB

Tanggal: 6 Oktober 2026, Asia/Jakarta. Baseline: commit `0910ca1`, HEAD `6b5e43d`; laporan `RUN_M4HUMAN_SMOKE` tetap menjadi bukti historis. Scope penelitian mengikuti `m4human_kinetok_v3`, dengan RPC sebagai input sensor, target offline, serta frozen encoder/motion/LLM sesuai tahapnya.

## Tujuan dan batas bukti

Perbaiki seluruh temuan audit pada jalur M4Human dan sambungan antartahap. Tingkatkan throughput training melalui batching, input pipeline, AMP yang benar, dan profiling terukur. Memenuhi VRAM bukan sasaran tersendiri: pilih konfigurasi tercepat yang memenuhi batas memori, stabilitas numerik, dan kualitas train/validation. Batch efektif, update budget, precision, dan keputusan gate dicatat sebagai bagian protokol; perubahan hardware tidak boleh diam-diam mengubah perlakuan C_base/C_kin.

Dataset nyata dan RTX 3060 berada pada perangkat lain. Pemeriksaan lokal membuktikan kontrak kode dan reproduksi bug, bukan speedup CUDA atau akurasi baru. Artefak run lama tidak ditimpa atau direkonstruksi dari angka ringkasan.

## Pembagian pekerjaan

| Jalur | Tugas | Verifikasi |
|---|---|---|
| Data | Exact-key coverage; metadata coverage; context per recording; LMDB/memmap/manifest aman untuk Windows spawn | Fixture key hilang, manifest interleaved, equivalence worker0/spawn, sensor-only |
| Encoder dan hardware | AMP overflow; resume early-stop; loader worker/pinning/prefetch; evaluasi tanpa record yang tidak dipakai; profiler batch/worker/precision | GradScaler overflow recovery, numerical/batch equivalence, deterministic sampler, bounded profiling |
| Gate dan artefak | Keputusan freeze terikat checkpoint; hash teks lintas newline; resource RSS yang benar; bundel run dapat divalidasi | Config gate tampering ditolak, arsip lengkap, byte/content hashing terdokumentasi |
| Integrasi | Perbaiki AMP dan input pipeline pada M/C/probe/projector; perbarui planning/runbook; audit diff | Semua contract/integration checks, pemeriksaan frozen weights, paired treatments dan provenance |

## Urutan pelaksanaan

1. Reproduksi dan perbaiki temuan correctness sebelum tuning hardware.
2. Tambahkan profiler terbatas pada train data; warmup dan timed steps terpisah, ukuran memori CUDA dan RAM eksplisit, OOM ditangani, kandidat tidak melebihi batas memori dan batch efektif yang dipilih peneliti.
3. Sediakan konfigurasi RTX 3060 yang eksplisit. Jangan menganggap batch terbesar paling cepat; ukur worker0/2/4, batch dan precision. Precision CUDA memerlukan parity/stability terhadap fp32.
4. Ikat gate pada metadata checkpoint. Smoke/debug tidak mendapat status ilmiah; pilot hanya dapat memenuhi gate bila kontrak/readiness predeclared. Threshold dipilih train/val sebelum selected run, dan config ekstraksi tidak boleh mempromosikan checkpoint.
5. Pastikan run dapat dibundel bersama checkpoint/prediksi dan witness; raw dataset tetap eksternal. Audit bundle semantik dari perangkat training tetap diperlukan.
6. Jalankan regression checks yang menangkap bug, seluruh suite, kemudian audit independen diff dan jalur nyata antartahap.
7. Terapkan perubahan yang telah diverifikasi ke workspace pengguna tanpa menimpa perubahan lokal lain. Laporkan kode yang selesai, pemeriksaan yang lulus, dan gate CUDA/data nyata yang belum dapat dijalankan lokal.

## Syarat sebelum training final di komputer dataset

- Audit ulang exact-key/metadata coverage, unit, kalibrasi, joint map, timestamp dan target anatomy.
- Jalankan profiler, tiny-overfit, fp32 versus AMP checks dan E pilot pada train/validation; pin konfigurasi dan gate dari bukti tersebut.
- Gunakan checkpoint yang merekam gate sah sebelum membuat state cache ilmiah.
- Jalankan M/probe quality gates sebelum menguji efek C_kin; C_base/C_kin serta projector conditions harus tetap matched.
- Kunci protokol test sebelum membuka held-out outcomes.

## Hasil implementasi dan verifikasi

Seluruh jalur pekerjaan di atas telah diimplementasikan dan diaudit oleh sub-agent data, training/hardware, gate/artefak, serta reviewer. Sembilan suite CPU contract/integration lulus; 55 source/config hashes tetap sama selama pemeriksaan. Laporan: [verification 6 Oktober](m4human_verification_20261006_hardening.json). Detail temuan, perbaikan dan batas bukti: [audit hardening](m4human_hardening_audit_20261006.md).

Profiler tersedia melalui `eksperimen_model.profile_m4human_encoder`; screening singkat dipisahkan dari profil konfigurasi terpilih minimal 100 successful timed updates untuk readiness. Tiny-overfit train-only, bundler tervalidasi, lima kandidat RTX3060 dan ekstraksi E/H/U batched tersedia. Ikuti [runbook](m4human_development_runbook.md#7-profiling-rtx-3060-12-gb-dan-gate-checkpoint) untuk komputer dataset. Perbaikan ini tidak menghasilkan benchmark CUDA atau hasil ilmiah baru pada laptop.

## Tambahan optimasi RAM16GB dan VRAM12GB

Permintaan memaksimalkan RAM/VRAM ditangani lewat empat jalur paralel: metadata/index worker yang ringkas; monitor RAM sistem/proses/worker dan headroom CUDA; profiler train-only downstream; serta audit independen yang membandingkan profiler dengan trainer aktual. Implementasi minimal memakai memmap/shared integer arrays, filesystem cache OS, psutil dan allocator native PyTorch. Tidak ada perubahan dimensi arsitektur atau tambahan input sensor untuk mengisi memori.

E sekarang dapat menyapu beberapa batch efektif eksplisit, termasuk16/64/128/256/512/1024, memilih throughput tertinggi di antara kandidat dengan parity, tanpa timed overflow dan reserve aman, lalu menulis config lengkap. M/C/probe/L mendapat profiler sendiri, dengan witness forward/loss/gradient, warmup/timed terpisah, median/p95 dan generation sweep untuk L. Perubahan batch efektif adalah protokol baru yang harus dibandingkan melalui pilot; learning rate tidak diskalakan otomatis.

Semua11 YAML memasang reserve RAM2048MiB dan headroom CUDA1024MiB. Kandidat E128×1, M32×1, C/probe16×1, L4×4 serta generation8 belum menjadi hasil benchmark. Monitor aktif dari persiapan hingga final validation/generation; kegagalan sampling/budget dan sampler yang gagal berhenti membuat run failed sebelum completion. RSS sum diberi semantik double-count halaman bersama; reserve RAM memakai available sistem. QA/H/U lazy dan index shared mengurangi salinan metadata pada spawn; hash satu file tensor diverifikasi dari bytes yang sama dengan deserialisasi.

Audit menemukan dan memperbaiki kesesuaian optimizer profiler/trainer, pemilihan generation config yang dikonsumsi loop L, normalizer readiness, shutdown monitor dan coverage persiapan dataset. Verifikasi terbaru: [dua belas suite CPU dengan source hashes](m4human_verification_20261006_hardware.json). [Runbook §7](m4human_development_runbook.md#7-profiling-rtx-3060-12-gb-dan-gate-checkpoint) memberi command sweep dan matched selection untuk komputer dataset.

E baseline efektif16 dipertahankan; kandidat128 adalah protokol berbeda yang harus dipilotkan, tanpa auto LR scaling. M32, C/probe16, L16 tetap matched. Real smoke evidence menunjuk complete validated E metrics dan exact SHA; resource evidence menunjuk bounded m4human measured-CUDA profile dengan matching config, parity/no-overflow/memory reserve. Hash readiness mengizinkan threshold train/val berubah sebelum selected run, sedangkan checkpoint full config mengunci threshold ekstraksi.

Historical smoke tidak diubah. Missing best/last/predictions berarti bundle masih incomplete; LF/CRLF compatibility hanya menyelesaikan representasi newline. Test nyata, tiny-overfit, real CUDA resources dan scientific outcomes belum dikerjakan laptop. Audit/source semantic, QA interval/support, RPC-only dan optional RT ablation tetap menjadi gerbang sebelum training final.
