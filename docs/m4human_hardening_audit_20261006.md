# Audit hardening M4Human — 6 Oktober 2026

Scope `m4human_kinetok_v3`, staging perubahan dari baseline commit `0910ca1`/HEAD `6b5e43d`. Ini audit implementasi dan kontrak, bukan laporan eksperimen M4Human atau benchmark RTX3060. `AGENTS.md`, README dan laporan smoke historis tidak diubah oleh pekerjaan dokumentasi ini.

## Status pemeriksaan

| Pemeriksaan | Status saat audit ini disusun | Batas bukti |
|---|---|---|
| Gate/artifact focused contract | Passed lokal | Synthetic fixtures termasuk metadata fake-real/fake-CUDA berlabel; bukan hasil nyata |
| Performance focused contract | Passed lokal | Loader/optimizer/parity wiring dan generated-report→gate interface |
| Final full contract/integration suite | Passed: 9/9 | [Laporan tersimpan](m4human_verification_20261006_hardening.json); CPU/synthetic, 55 source hashes tidak berubah selama pemeriksaan |
| Bounded same-batch AMP retry dan lazy validation records | Passed lokal | Numerical replay E/M/QA, masked loss, frozen weights, dan integrasi; CUDA aktual belum diuji |
| Tiny-overfit fixed train subset | Implemented; actual data not_run | Diagnostic loss/parity, tidak menulis scientific checkpoint |
| Windows/RTX3060 real-data profile | Not_run lokal | Workers/memory reserve/parity/throughput harus diukur perangkat tujuan |
| Dataset semantic audit dan scientific training | Not_run lokal | Joint/unit/calibration/source target semantics tidak dibuktikan fixture |
| Held-out outcomes dan uncertainty | Not_run | Threshold/policy dipin train/val sebelum test lock |

Suite selesai pada 6 Oktober 2026, 11:43 UTC, dengan sembilan pemeriksaan lulus dalam sekitar 54 detik. Environment lokal memakai PyTorch 2.14.1 dan Transformers 4.45.2; versi dependency lengkap dan output setiap pemeriksaan tersimpan di laporan. Komputer smoke historis memakai PyTorch 2.6/CUDA, sehingga suite perlu dijalankan lagi pada environment training tersebut. Pemeriksaan tambahan: syntax 42 modul M4Human, parsing 11 YAML, dan `--help` delapan CLI baru/terkait lulus. Laporan historis tetap utuh.

## Temuan dan perbaikan

Data boundary memakai exact source-key/metadata coverage, konteks per recording/segment, lazy per-process LMDB dan memmap untuk Windows spawn, serta epoch state yang diteruskan pada persistent workers. RPC XYZ/intensity menjadi input P0; annotation dibuka eksportir/target reader offline, bukan forward sensor. RT optional ablation bukan asumsi Doppler.

Loader/runtime memaparkan workers/pinning/nonblocking/prefetch/spawn dan evaluation/extraction batching. Profiler train-only memisahkan fitting normalizer dan warmup dari timed updates, mencatat successful updates, timed overflow, CUDA allocated/reserved/budget/reserve, parity forward/loss/backward serta rekomendasi matching config. Tidak ada measured latency/VRAM RTX3060 dari laptop. E16 baseline dipertahankan; kandidat E128 adalah protokol baru tanpa automatic learning-rate scaling. M32, C/probe16, L16 tetap efektif sama dan paired treatments matched.

AMP memakai fp32 reductions/physical loss, scaler unscale sebelum clipping dan pengecekan seluruh optimizer gradients. Overflow tidak memajukan successful-update/scheduler/exposure; batch yang sama dicoba ulang secara bounded. Norm clipping C tetap terpisah pada compressor+fidelity dan auxiliary; telemetry bernama `compressor_fidelity`/`auxiliary` before/after tersedia pada loop C yang mengisi `norms_out`. Scalar maximum return helper sendiri bukan norm per grup. Frozen weights/checkpoint integrity tetap dicek. Final regression menentukan klaim implementation checks, bukan quality evidence.

L mendukung native SDPA dan wrapper activation checkpoint eksplisit ketika projector training. Frozen Qwen tetap eval dan input gradient tetap mengalir. Tiny Qwen berbobot identik lulus kesetaraan eager/SDPA, checkpoint loss/gradient dan generation batched/singleton pada CPU. Pretrained Qwen dan CUDA fit belum diukur. OOM batch generation melepaskan traceback sebelum fallback singleton; fixture menguji pemulihan peer dan urutan ID.

Resume M/C/probe menandai akhir epoch pada exhaustion alami walaupun batch terakhir tanpa supervisi; budget yang menghentikan iterator di tengah epoch tetap incomplete. Regresi memakai loop training nyata pada fixture. Ekstraksi E, H dan kedua U lulus batch1/batch4 equivalence: ID, lineage, mask dan timestamp sama, float toleransi 3e-6. Ekstraksi E/H/U untuk inferensi tetap dapat berjalan tanpa annotation target.

Encoder gate version2 mengikat mode, full resolved config, root threshold, source/sensor policy, verified readiness artifacts dan selected decision. Threshold root dideklarasikan dari train/val sebelum selected run; tidak dipilih dari test. `real_smoke_passed` dan `resource_profile_passed` adalah mapping `artifact_path`+exact `artifact_sha256`; boolean/invented SHA ditolak. Smoke evidence memerlukan complete validated E run m4human, finite nonzero update/root support dan source/schema/joint/coordinate/target/split/recipe lineage cocok. Resource evidence memerlukan m4human measured CUDA parity, selected settings match setelah internal normalizer dikeluarkan, nol overflow, reserve aman,100+ complete timed updates dan exact timed exposure. Sweep10-step hanya screening dan ditolak sebagai readiness final. Readiness hash memakai config tanpa evidence/runtime-derived normalizer/lineage dan root threshold; full checkpoint config tetap mengunci threshold. Jadi threshold baru sebelum run dapat ditentukan train/val, tetapi ekstraksi tidak dapat mempromosikan checkpoint lama lewat config. Smoke selalu non-scientific; pilot membutuhkan explicit predeclaration+readiness.

Checkpoint extraction memverifikasi recorded spec/decision dan config tanpa membuka lagi readiness files atau raw annotations. Legacy debug masih usable dengan `--allow-debug`, metadata cache non-scientific. Verified smoke/profile artifacts dapat tetap berada di komputer training; checkpoint harus portable untuk inferensi, dengan lineage/hash utuh.

## Artefak dan laporan historis

Metrics/history completion baru memakai `utf8_lf_sha256` (UTF8, CRLF→LF saja). Marker legacy menerima LF/CRLF byte-hash equivalence dengan isi identik; perubahan konten atau lone-CR tampering ditolak. Source/checkpoint/cache lineage tetap SHA256 exact bytes. RSS current berbeda dari true peak; Windows peak memakai `peak_wset` jika tersedia, null jika tidak. CUDA peak per-device dicatat.

Validator streaming prediction records memeriksa ID/coverage dan merekonsiliasi sums/counts; QA scorer mempertahankan indeks yang dibutuhkan scoring. Package run memvalidasi sumber dan ZIP yang diekstrak, dengan checksums exact bytes dan best/last/predictions/witness/diagnostics. Dataset/raw targets tidak dibundel default, optional audit metadata hanya file teks kecil. QA reference harus relative dalam run agar portable. Artifact integrity tidak sama dengan scientific validity.

`docs/report_training/RUN_M4HUMAN_SMOKE` yang tersedia kehilangan `best.pt`, `last.pt`, dan `predictions_val.jsonl`; bundle ini tetap incomplete. Hash newline compatibility tidak menutup omission tersebut. Historical completion/metrics tidak ditulis ulang, missing arrays/models/predictions tidak direkonstruksi dari angka report.

## Sebelum final training

Ikuti [runbook §7](m4human_development_runbook.md#7-profiling-rtx-3060-12-gb-dan-gate-checkpoint): audit paket aktual, smoke complete, train-only tiny-overfit, pilot train/val, profile CUDA pilihan, predeclare threshold dan simpan actual evidence hashes. Candidate YAML tetap unmeasured sampai report perangkat dataset ada. Kunci physical/language test protocol sebelum held-out outcomes. QA `task_interval_s` adalah interval query, `label_support`/`evidence_support` adalah frame valid recipe; nominal interval tidak membuktikan semua frame valid.

Referensi implementasi: [AMP PyTorch 2.6](https://docs.pytorch.org/docs/2.6/amp.html), [DataLoader PyTorch 2.6](https://docs.pytorch.org/docs/2.6/data.html), [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). Panduan tuning mendukung opsi worker/pinning/checkpoint; benchmark aktual tetap menentukan konfigurasi, bukan tutorial.

## Pemeriksaan praktis runbook/source staging

Entry point/run flags diperiksa terhadap parser aktual: E trainer `--output-dir/--mode/--device/--precision`, E cache `--batch-size` fp32, profiler fixed-batch divisors, tiny-overfit train-only, M/C/probe condition, H/U extraction output/config dan validator positional run-directory. H/U device/batch ditetapkan config, bukan CLI yang tidak tersedia. M/C/probe `scientific_gates_passed` dan `precision_gate_passed` perlu diselesaikan melalui bukti actual gate; false/null templates tidak runnable final dengan sendirinya.

Pembacaan ulang `original_hashes.json` mencocokkan108 existing-file hashes pada source checkout yang dicatat manifest: tidak ada file existing berubah/hilang ketika audit praktis dilakukan. Ini pemeriksaan read-only terhadap baseline lokal sebelum apply staging; tidak membuktikan untracked files atau hasil suite final. Source manifest tidak ditulis ulang. Kandidat batch tetap unmeasured; tidak ada status scientific CUDA yang dikarang oleh dokumentasi.

Tambahan sustained resource gate: selected config direprofile dengan warmup5/timed100 dan satu microbatch/worker/precision pilihan ke `profile_E_selected100.json`; evidence menunjuk report tersebut. Warmup skips dapat tercatat, timed updates/exposure harus lengkap. Candidate L worker/prefetch bukan fitur loop QA direct microbatch saat ini; optimasi L yang aktif adalah batching, native SDPA, explicit checkpoint dan generation batch.

## Audit tambahan RAM16GB/VRAM12GB

Metadata H/U/QA sekarang lazy; UID index memakai16bytes per baris untuk hash+row index, offset dan array integer berbagi storage saat spawn. Hash collision diverifikasi terhadap ID lengkap, termasuk duplicate rejection. Fixture5000 baris berprovenance panjang mempertahankan transport metadata spawn di bawah16KiB; angka ini hanya fixture, bukan pengukuran corpus M4Human. File tensor window mempunyai bound64MiB dan satu pembacaan yang sama dipakai untuk SHA serta `weights_only` deserialization. Dataset sumber tidak ditulis.

Monitor psutil mencatat parent+recursive descendants RSS/USS, available RAM sistem, swap delta dan CUDA free/allocated/reserved/peak/headroom. RSS sum bisa menghitung halaman bersama berulang dan bukan penggunaan fisik. USS nullable bila OS tidak menyediakan akses; reserve memakai available RAM yang nyata. Sampling250ms tidak menjamin melihat setiap lonjakan di antara poll; overhead dan scope dicatat. Shutdown sampler yang melewati timeout gagal tertutup. Dua fixture loop training M menunjukkan pelanggaran saat fetch maupun final context-exit menghasilkan metrics/completion failed dan logger closed.

E profiler memonitor sejak konstruksi dataset, menyapu beberapa effective batches eksplisit dan menulis full config dengan policy/hash konsisten ketika direprofile. M/C/probe/L profiler tidak menulis checkpoint ilmiah atau membaca validation/test samples. Optimizer grouping sesuai trainer aktual, exposure memakai successful logical updates, timed retry disqualifies. L memeriksa frozen weights/gradients, memakai trainQA1 untuk reference fp32, dan hanya mengekspor selected config bila generation kandidat juga aman. Config generation memakai field yang sama dengan trainer. Native SDPA/checkpoint tidak memberi bukti semua microbatch/QA akan muat; real Qwen tetap perlu diukur.

Normalizer readiness kini memeriksa finite mean/std4, std positif, exact canonical hash sesuai normalizer training dan lineage; artefak lama yang normalizernya berbeda/tidak tercatat ditolak. Verified spec menyimpan normalizer hash untuk pemeriksaan checkpoint portable. Normalizer/runtime tidak dimasukkan kembali ke readiness config hash; hasil profil terikat lewat evidence hash dan verifikasi normalizer eksplisit.

Verifikasi tambahan lulus **12/12 suite** dalam56.46detik, dengan59 source/config hashes tidak berubah selama pemeriksaan: [laporan tersimpan](m4human_verification_20261006_hardware.json). Pemeriksaan CLI help lima entrypoint terbaru serta syntax46 modul/YAML11 juga lulus. Cakupan termasuk memory transport, resource-budget training failures dan stage performance fixtures. Ini CPU dengan fixture, bukan benchmark CUDA atau bukti kualitas baru. Fixture CPU memakai reserve kecil yang dinyatakan eksplisit;11 config produksi tetap RAM2048MiB/CUDA1024MiB. Config kandidat baru E128×1, M32×1, C/probe16×1, L4×4/generation8 harus diprofilkan pada perangkat dataset; hanya E mengubah batch efektif baseline16 dan wajib pilot protokol baru. Kedua perlakuan C/L dan probe pembanding harus memilih kombinasi umum yang feasible.
