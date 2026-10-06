# 05 — Evaluasi Pelestarian Kinematik pada Token Radar dan Pemanfaatannya oleh Frozen LLM

Tanggal revisi total: 6 Oktober 2026. Kontrak bersama: `m4human_kinetok_v3`.

Dokumen ini mengoperasionalkan [proposal riset terbaru](../proposal_riset_terbaru.md), [encoder](01_encoder.md), [motion model dan tokenizer](02_dynamic_model.md), [decoder dan target kinematik](03_decoder.md), serta [LLM layer](04_llm_layer.md). Ini adalah **rencana implementasi dan pengujian hipotesis**, bukan laporan eksperimen. Tidak ada akses ke dataset perangkat lain, hasil training, pengukuran GPU, atau keunggulan metode yang dinyatakan telah terbukti.

Processed M4Human berada di perangkat eksperimen lain menurut pengguna; root rencana ialah `/dataset`. Path LMDB, isi paket, timestamp, kalibrasi, dan joint map harus diaudit di perangkat tersebut. Artefak MM-Fi dan eksperimen forecasting sebelumnya tetap historis: bukan bukti bagi kontrak baru.

## 1. Pertanyaan penelitian dan batas kesimpulan

Penelitian ini mengembangkan dan menguji **kinematic-preserving motion tokenizer**: modul yang membentuk token kontinu ringkas dari representasi gerak radar, dengan supervisi yang menjaga posisi dan perubahan kecepatan pada **token akhir yang benar-benar diterima projector**. Membuat tokenizer maupun menempelkan kinematic loss pada decoded tokens bukan kebaruan dengan sendirinya. Kontribusi kandidat ialah kombinasi metode adaptasi untuk radar M4Human, pengukuran keterbacaan token final di bawah budget terbatas dengan kontrol yang kuat, dan pengujian pemanfaatannya oleh frozen LLM. Ini bukan klaim first tokenizer, first output-side physical loss, atau first token-to-language.

Empat pertanyaan harus dibedakan:

| Pertanyaan | Perbandingan dan bukti | Batas kesimpulan |
|---|---|---|
| RQ1 — Keterbacaan setelah kompresi | Probe independen pada memori kaya `M/H` versus `U_base`; `Z` sebagai diagnostic tambahan; beberapa budget bila feasible | Apakah ada penurunan keterbacaan target terpilih pada setup ini; bukan bukti semua kompresi pasti merusak |
| RQ2 — Pelestarian pada token akhir | `U_kin` versus `U_base` dengan backbone, compressor, full-H fidelity, label, dan budget yang dikontrol | Efek strategi penempatan supervisi kinematik pada output kompresi; bukan keunggulan arsitektur murni |
| RQ3 — Pemanfaatan oleh frozen LLM | Dua projector matched untuk `U_base/U_kin`, skor QA dan kontrol grounding | Apakah perbaikan keterbacaan diikuti manfaat downstream; bukan bukti LLM memahami seluruh fisika |
| RQ4 — Batas kualitas, budget, biaya dan held-out | K16 primer; K8/K32 secondary pradefinisi bila feasible; token/precision/latency, hasil per task/subjek held-out, serta uncertainty berkelompok | Trade-off dan batas pada cohort/perangkat yang diuji; satu budget tidak membuktikan kurva multi-budget, satu seed tidak membuktikan robustness |

Mapping RQ4: aturan budget pada §3, split held-out pada §5, metrik/uncertainty pada §10, dan resource aktual versus estimasi pada §13. Pengujian biaya tidak digantikan oleh hitungan ukuran tensor atau besar download dataset.

**Hipotesis utama:** pada budget sama, supervisi kinematik yang mencapai token akhir dapat meningkatkan keterbacaan informasi gerak dibandingkan supervisi tambahan yang hanya melekat pada memori sebelum kompresi. **Hipotesis downstream terpisah:** frozen LLM dapat memanfaatkan peningkatan tersebut. Keduanya boleh mempunyai hasil berbeda.

Batas lingkup ialah **kinematika gerak teramati**: posisi joint/root, velocity, perubahan speed, dan relasi bagian tubuh yang labelnya reliable. Urutan onset hanya masuk klaim jika panel order tersendiri lulus audit. Tidak ada target gaya, torsi, kontak terukur, COM, niat, diagnosis keseimbangan, hukum Newton penuh, atau forecasting wajib. Pelvis bukan otomatis COM. Dataset berukuran sekitar 50 GB bukan bukti kecukupan sampel independen.

### Dasar keputusan dan status pengetahuan

| Label | Arti |
|---|---|
| **F** | Fakta dari sumber primer yang diperiksa; tidak menjamin isi paket lokal |
| **T** | Penurunan matematis dengan asumsi eksplisit |
| **R** | Keputusan metodologis/desain dengan alasan |
| **H** | Hipotesis/default kandidat, belum divalidasi empiris |

**F:** RadarLLM menunjukkan kelayakan trained radar-to-text, bukan pemahaman sensor universal atau keberhasilan projector-only pada frozen LLM. Token tetap berurutan (`L=T/r`); codebook size berbeda dari panjang token. Appendix H menunjukkan ROUGE-L 29,8→31,2 dengan VQ pada radar-to-text-only, sementara METEOR turun. Paper tersebut tidak menyediakan ablation kinematik yang mengisolasi kehilangan posisi/velocity akibat budget kompresi. Jadi kuantisasi tidak diasumsikan selalu merusak. [RadarLLM, metode dan Appendix H](https://arxiv.org/html/2504.09862v2).

**F:** HuMoCon membahas hilangnya detail frekuensi tinggi dan memakai velocity reconstruction. Penghapusan objective pendukungnya menurunkan skor BABEL-QA 0,711→0,637; ini bukan ablation jumlah token. Menambahkan velocity loss saja tidak cukup untuk klaim kebaruan. [HuMoCon, Table 3](https://arxiv.org/html/2505.20920v1).

**F:** FiGMo, preprint Juni 2026, melaporkan penurunan akurasi 0,758→0,711 saat timestamp grounding dihapus. Pentingnya informasi waktu bukan bukti bahwa semua tokenizer menghapus urutan. [FiGMo, §4.4](https://arxiv.org/html/2606.20888v1).

**F:** SeMoCo, preprint Agustus 2026, merekonstruksi gerak dari quantized tokens dan melatih decoder dengan joint-position, velocity dan acceleration losses. Output-side kinematic supervision sudah mempunyai pendahulu langsung; kontinuitas token dan penggunaan radar saja tidak otomatis membuat prinsip loss tersebut baru. [SeMoCo, §3.2](https://arxiv.org/html/2608.24334v2).

**F:** MoTok bukan hanya motion generation: Appendix A.2/D.3 menguji motion-to-text dan melatih captioner atas tokens dari tokenizer frozen. Itu menjadi preseden token→language dan perbandingan downstream. Frozen tokenizer dengan captioner yang dilatih berbeda dari fully frozen pretrained LLM dengan projector-only, tetapi tidak boleh dianggap tidak pernah ada penggunaan token untuk bahasa. [MoTok, Appendix A.2/D.3](https://arxiv.org/html/2603.19227v1).

**R:** Literatur memotivasi pertanyaan; tidak membuktikan bottleneck pada M4Human kita. Hasil caption, reconstruction loss, probe, dan QA adalah bukti berbeda. Kebaruan kandidat berada pada formulasi/adaptasi/protokol pembuktian gabungan yang spesifik, bukan nama modul atau satu loss yang sudah dikenal. Klaim metode yang benar-benar baru memerlukan pembandingan lebih dekat dengan pendahulu tersebut; hasil positif P0 tetap dapat mendukung manfaat adaptasi tanpa membuktikan originalitas prinsip. Klaim novelty memerlukan penelaahan literatur yang mutakhir, bukan hanya hasil mengungguli baseline lokal.

## 2. Rantai representasi yang benar-benar diuji

Seluruh nama model di bawah adalah **nama desain implementasi lokal**, bukan model opensource yang sudah dirilis dengan nama tersebut. Nilai referensi adalah **H** sampai audit dan smoke test.

```text
Radar observed window, sensor masks, timestamps
  -> encoder bersama dan frozen
  -> motion backbone bersama dan frozen
  -> H[B,32,23,128]  : joint-time representation kaya
     M[B,32,23,256] : concat(H_tj, H_tglobal), memory untuk compressor/probe
     Z[B,32,256]    : masked mean 22 joint concat global, diagnostic ringkasan
  -> compressor terlatih, C_base atau C_kin
  -> U[B,K,256] + token_mask + bin times/provenance
  -> projector kondisi bersangkutan
  -> P[B,K,d_llm]
  -> frozen Qwen + pertanyaan -> jawaban terstruktur
```

**R:** `M` diturunkan dari `H`; `Z` bukan satu-satunya sumber pembentuk `U`. Oleh sebab itu, selisih `Z`–`U` saja tidak membuktikan kehilangan akibat kompresi. Kandidat tokenizer mempunyai akses ke memory joint-time yang sama, bukan hanya ringkasan joint-pooled.

**T:** Untuk model yang sudah fixed dan metadata sensor `S` yang sama, `U=f(H,S)` tidak dapat menambah informasi baru tentang target yang tidak tersedia dalam `H,S`. Ini tidak mewajibkan informasi target terpilih hilang: kompresi dapat membuang redundansi. Probe mengukur keterbacaan oleh keluarga readout tertentu, bukan mutual information atau invertibilitas penuh.

### Kontrak tensor dan support

| Bagian | Referensi kontrak `m4human_kinetok_v3` |
|---|---|
| Input radar | `RPC[B,32,4,512,4]`: 32 anchor, context kronologis causal 4, 512 points, XYZ/intensity **jika schema terverifikasi** |
| Encoder | `M4HumanSetEncoderV2`, scratch, shared point MLP+pool dan context temporal; satu checkpoint bersama |
| Encoder outputs | `P_enc[B,32,22,3]` root-relative meter; `r_enc[B,32,3]` global pelvis meter; `f_enc[B,32,256]` |
| Motion backbone | `CausalDSTformerLiteV1`, scratch, 4 stage, width128, 23 spatial tokens, temporal causal; bukan autonomous transition model |
| Representasi kaya | `H[B,32,23,128]`; node terakhir adalah global; `M_tj=concat(H_tj,H_tglobal)`, termasuk query global |
| Ringkasan diagnostic | `Z[B,32,256]=concat(masked_mean_joints(H_tj),H_tglobal)` |
| Compressor | `KinematicTokenLearnerV1`: K bin kronologis, satu token kontinu/bin, query cross-attention bin-restricted, width256, 4 heads, 1 block + LN/FFN |
| Budget | `K=16` primer; `K=8,32` secondary pradefinisi; K berarti token budget, bukan 23 spatial nodes atau codebook size |
| Fidelity decoder | Time+joint query decoder dari seluruh U, width256, 4 heads, output128/query untuk rekonstruksi seluruh `H[32,23,128]` |
| Auxiliary physical decoder | Time+joint queries, posisi dan velocity joint/root untuk seluruh observed window; arsitektur/parameter identik kedua kondisi |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct`, pretrained, seluruh bobot frozen; model/tokenizer revision dipin |
| Projector | Scratch LN256→Linear512→GELU→Linear1536; satu projector independen per kondisi |
| Default numerik | fp16 AMP + GradScaler jika smoke lulus; physical targets/loss/derivatives fp32; bf16 alternatif yang harus diuji |

**F:** Config publik Qwen mencantumkan hidden size 1536; implementasi tetap membaca revision yang dipin. [Config resmi Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json).

Bin dibentuk dari bounds frame/waktu sumber yang tetap, bukan dari label, kategori jawaban, atau joint ground truth. Query compressor tidak bergantung pertanyaan. Timestamp/identitas joint sensor-derived dipakai dengan policy sama. Empty bin di-zero dan mask false, termasuk sesudah encoding; all-invalid window ditangani eksplisit. Tidak ada `f_enc`, raw predicted state, GT, atau jawaban yang menyelinap langsung ke `U`/projector.

**R:** Temporal trunk causal sampai setiap anchor; **kompresi dan reconstruction dari U seluruh window adalah retrospective**. Readout untuk t dapat menggunakan seluruh observed window, termasuk anchor sesudah t. Ini bukan kebocoran masa depan di luar window, tetapi tidak boleh disebut online-causal readout per-timestep. Projector/LLM hanya melihat window observasi, bukan frame forecast.

## 3. Perbandingan primer: C_base versus C_kin

Perbandingan utama tidak lagi `C_history` versus `C_dynamics`. Keduanya dahulu mengubah akses predicted state, kapasitas, dan supervisi upstream sekaligus. Bila tetap dibahas, statusnya historical/secondary dengan confounds, bukan bukti pelestarian token.

### Apa yang disamakan

| Komponen | C_base | C_kin |
|---|---|---|
| Radar/encoder/motion/cache | Sumber sensor, checkpoint H/M, masks, waktu dan IDs yang sama | Sama |
| Compressor | Learned query tokenizer, K×256 | Arsitektur, parameter, initialization, dan K yang sama |
| Fidelity target | Seluruh frozen H, time+joint decoder dari U | Target, decoder, normalizer, masks dan loss yang sama |
| Label kinematik tambahan | Posisi/velocity GT, support dan scale yang sama | Sama; tidak memperoleh anotasi baru |
| Auxiliary physical head | Time+joint query head atas **M** | Head berparameter sama atas **U** |
| Gradien auxiliary | M frozen: update auxiliary head saja, **tidak mencapai compressor** | Update auxiliary head dan compressor |
| Gradien fidelity | M/H target detached; fidelity meng-update decoder dan compressor | Sama |
| Projector | Projector scratch sendiri | Projector scratch sendiri, initialization matched |
| Alignment QA | IDs, prompts, target, budget update/exposure yang sama | Sama |

**R:** Treatment yang disengaja adalah **lokasi keterikatan supervisi fisik**: memori kaya sebelum kompresi versus keluaran token sesudah kompresi. Backbone upstream sudah mendapat supervisi fisik bersama dan kedua kondisi mengakses label sama. Tidak boleh menulis bahwa semua jalur gradien, intensitas supervisi yang mencapai compressor, panjang memory auxiliary, atau biaya compute persis sama. Sumber auxiliary berubah dari M menjadi U karena itulah treatment; ini bukan isolasi arsitektur baru semata.

C_base tetap baseline terlatih yang kuat: loss fidelity-nya merekonstruksi **seluruh joint-time H**, bukan hanya Z yang telah joint-pooled, dan bukan compressor random. Kedua compressor menerima memory H yang sama. Readout auxiliary M berguna sebagai kontrol keterbacaan upstream; ia tidak dipresentasikan seolah loss tersebut melatih C_base.

### Objective stage C

```text
M_raw, H_target = fixed common cache
M_attn = build_rich_attention_memory(M_raw, sensor_time, sensor_masks)
U_condition = compressor_condition_from_memory(M_attn)
H_hat_norm = fidelity_decoder_condition(U_condition, time_joint_queries)

H_target_norm = (stopgrad(H_target) - mu_H[channel]) / s_H[channel]
L_H = mean_valid((H_hat_norm - H_target_norm)²)

C_base:
  kin_hat = auxiliary_head_base(M_attn, time_joint_queries)
  L_total = L_H + lambda_aux * L_kin(kin_hat, GT_position_velocity)

C_kin:
  kin_hat = auxiliary_head_kin(U_kin, time_joint_queries)
  L_total = L_H + lambda_aux * L_kin(kin_hat, GT_position_velocity)
```

**R/H:** `mu_H` ialah mean channel train-only; `s_H` ialah population standard deviation (ddof=0), floor 1e-3, pada shared eligible H. Decoder mengeluarkan H_hat_norm; inverse-transform untuk metrik raw H mengikuti dokumen 03. Invalid nodes tidak masuk statistik, dan normalizer tidak dihitung terpisah per kondisi. Loss fidelity memakai valid joint/global entries yang sama dan target detached. `Z` reconstruction hanya boleh menjadi secondary diagnostic/ablation, bukan menggantikan fidelity utama.

`L_kin` mengikuti empat kelompok, units dan masks [dokumen decoder](03_decoder.md): posisi joint-relative, posisi root-global, velocity joint-relative, velocity root-global. Reference component MSE:

```text
L_kin = L_p_joint + L_p_root + 0.5 * (L_v_joint + L_v_root) / 2
position error scale = 1 metre
velocity error scales = audited train-only RMS components
```

Pelvis-relative constrained-zero dikeluarkan dari relative loss. Output auxiliary berupa posisi/velocity langsung, tanpa residual predicted-pose atau raw-state skip; query berisi identitas waktu/joint, bukan GT coordinates atau kategori jawaban. **Time/joint reconstruction queries sah** karena diturunkan dari support dan skeleton registry, bukan nilai target. Root-global dan relative joint disatukan hanya dengan operasi `p_global=p_relative+r`, `v_global=v_relative+v_root`. Loss tidak menerima label jawaban QA pada P0. Posisi/velocity tambahan pada auxiliary tetap target yang sama, bukan anggaran anotasi tersembunyi.

Nama bobot kanonis adalah `lambda_aux`, konsisten dengan dokumen 02/03; default kandidat **H** ialah 1,0, bukan nilai optimal yang telah terbukti. Bobot final, fidelity weighting, schedule dan checkpoint rule dipilih train/validation dengan ruang tuning yang dicatat sebelum test. Aturan selection dan anggaran tuning berlaku sama. Catat normalized fidelity validation dan physical diagnostics terpisah; jangan memilih proposed dengan kriteria yang dibuat setelah test. Auxiliary baseline atas M dapat mempunyai loss lebih rendah karena inputnya lebih kaya: total training loss antar kondisi bukan ukuran pelestarian U.

M pada cache tetap concat H mentah; M_attn adalah view dengan PE waktu sensor yang ditambahkan tepat sekali oleh helper bersama dokumen 02. Compressor, auxiliary baseline, dan probe M memakai view/mask yang sama; tidak ada tambahan joint embedding pada memory. Query decoder memiliki joint embedding sendiri, sedangkan U akhir tidak diberi encoding memory ulang. Clipping tahap C dipisahkan antara grup compressor+fidelity decoder dan auxiliary head (masing-masing norm 1). Initializer q0 Normal(0,0,02), tanpa weight decay, serta initial-state checksum semua grup wajib tercatat.

### Makna budget dan kontrol tambahan

K16 primer tidak diganti menjadi K lain karena hasil test lebih menarik. K8/K32 diperlukan untuk klaim **trade-off lintas budget**, tetapi keputusan kelayakan seluruh pasangan ditetapkan pada pilot sebelum test. Bila hanya K16 feasible, laporkan hasil satu budget dan hapus klaim multi-budget.

Di dalam setiap K, parameter compressor/decoder/query, initialization, minibatch order, total successful updates dan data exposure matched. Lintas K, jumlah query/compute/panjang token dapat berubah; laporkan parameter counts aktual, bukan menganggap perubahan K mengisolasi kompresi sempurna. K32 menghilangkan downsampling temporal per-bin, tetapi masih mengompresi spatial nodes dan width, jadi bukan otomatis tanpa kompresi.

`C_pool` (masked mean Z per-bin) opsional untuk diagnosis mekanisme pooling; bukan control primer karena architecture/loss berbeda. `C_state` opsional jika mengklaim manfaat melampaui koordinat predicted eksplisit. Mengalahkan baseline lokal tidak sama dengan mengalahkan RadarLLM atau SOTA; tanpa reproduksi matched, literatur hanya pembanding konsep.

## 4. Tahapan, training ownership, dan paket minimum

| Stage | Dilatih | Dibekukan | Gate keluaran |
|---|---|---|---|
| E — sensor | Encoder + pose/root heads | LLM tidak dipakai | Satu encoder common divalidasi, freeze, sensor-only cache |
| M — motion | Trunk + full-H physical decoder bersama | Encoder | Satu motion checkpoint common, p/v support reliable, H/M/Z fixed |
| C — compression | Dua compressor, fidelity decoder, auxiliary head sesuai routing | Encoder, trunk dan stage-M decoder | U_base/U_kin, reconstruction dan gradient-route tests |
| Probe | Readout/probe independen saja | Seluruh representasi yang diuji | Held-out readability M versus U; Z secondary |
| L — alignment | Projector kondisi bersangkutan saja | Encoder, trunk, compressor, seluruh physical heads, Qwen | Dua projector matched; no LoRA dan no QA gradient upstream |
| Test | Tidak ada | Semua checkpoints dan recipe | Prediksi seluruh locked cohort dan laporan |

Encoder Point-MAE transfer bukan requirement P0: hanya tambahan setelah audit channel/noise/checkpoint compatibility. Struktur joint–waktu mempunyai preseden MotionBERT, tetapi penggunaan bobotnya tidak otomatis kompatibel dengan radar. [MotionBERT, ICCV 2023](https://arxiv.org/abs/2210.06551).

P0 minimum:

1. Audit data/split/targets dan synthetic tests.
2. Satu encoder dan motion backbone common, dengan sanity/smoke/tiny-overfit.
3. C_base dan C_kin pada K16, full-H fidelity yang sama, auxiliary routes eksplisit.
4. Probe physical p/v serta dua core task: `root_speed_trend` dan `relative_limb_motion`.
5. Majority/text-only, rule dari predicted states, dan probe tanpa LLM.
6. Dua matched projector, QA terstruktur, serta donor nyata berlabel jika cohort memadai.
7. Test lock, grouped uncertainty, biaya aktual dan batas hasil.

Menyatakan hipotesis AI negatif tidak mensyaratkan memakai banyak backbone atau sweeping semua model. Sebaliknya, run gagal belajar karena bug/target buruk tidak boleh dilaporkan sebagai bukti negatif ilmiah.

## 5. Audit dataset, split, dan provenance

**F:** Repository M4Human menyediakan processed radar dan jalur annotation/kalibrasi; paket processed modality radar disebut sekitar 50 GB, bukan ukuran RPC sendiri yang telah diaudit. Kode utils menunjukkan body-parameter preprocessing dan temporal context. Fakta publik itu bukan audit paket `/dataset` pengguna. [M4Human repository](https://github.com/FanJunqiao/M4Human), [utils resmi](https://github.com/FanJunqiao/M4Human/blob/main/dataset/m4human_utils.py).

Audit wajib, read-only:

1. Path/tata letak LMDB, codec/schema, frame keys, recording IDs, subject/action map, index inventory, missing/duplicate entries, source commit.
2. XYZ/intensity order, satuan, finite/padding mask dan intensity normalization train-only. **Tidak mengasumsikan Doppler atau SNR**. Tidak memakai annotation bbox untuk crop inference.
3. Kalibrasi radar–annotation, sumbu/origin, offset z dan inverse transform. Jangan menerapkan shift dua kali. `trans` bukan otomatis pelvis; 22 joints/pelvis index berlaku setelah map terverifikasi.
4. Timestamp asli, pairing radar–MoCap, dropped/repeated frames, non-increasing time, gap dan nominal rate. Grid nominal 12 Hz hanya asumsi jika waktu aktual tidak tersedia; batas klaim temporal dicatat.
5. Sensor-derived validity dipisah dari annotation validity. Shape/gender/params hanya boleh dipakai membentuk target offline; bukan input encoder/compressor/LLM.
6. RPC-only reader diuji dengan akses params/RGB/depth/action/GT diblok; subject/action hanya untuk sampling/split/evaluasi.
7. Shared cache H/M/Z membawa context/source frames, local window reset, dtype, model/schema/joint/time hashes. Cache partial atau full-recording yang berbeda support tidak menggantikan cache window32.

**R:** LMDB immutable dibuka read-only, handle per process, mulai workers0. `readonly=True` tidak otomatis berarti lock file tidak berubah. `lock=False` hanya pada immutable snapshot tanpa writer concurrent; jika belum terjamin, selesaikan snapshot/locking sebelum eksperimen. [python-lmdb](https://lmdb.readthedocs.io/en/release/).

Split held-out subject ditetapkan **sebelum seluruh training**; semua recording subject test tetap held-out bagi encoder, trunk, normalizer, threshold, probe, projector dan recipe tuning. Context/derivative/window dibentuk setelah split; tidak ada union support lintas recording, gap, atau partition. Official index dipakai hanya setelah disjointness/audit, bukan mengarang N_subject.

Context4 membaca t−3..t. Window motion memuat 32 anchor; cold-start union support dapat memuat 35 frame radar. Pada asumsi uniform12Hz, span anchor 31/12≈2,583 detik dan sensor union 34/12≈2,833 detik. Ini **T**, bukan latency compute atau hasil audit timestamps. Seluruh dukungan target/probe/QA pada window sama.

### Test lock

Pilih melalui train/validation, lalu hash sebelum membuka test: task panel, thresholds, joint map/derivative, normalizers, sampling/stride, K primer/secondary, tuning budget, trainable map, checkpoint rules, query/readout capacity, prompts/parser/generation, metric classes/weights, effect tolerance dan grouped uncertainty policy.

`test_contract.json` mencatat IDs/splits/support, class registry, exclusion policy, model/cache/recipe/config hashes, selected claims dan mandatory controls. Schema/integrity test boleh diaudit; target outcome test tidak memilih hyperparameter. Perubahan sesudah melihat hasil test diberi label exploratory, bukan test konfirmatori yang sama.

## 6. Target kinematik dan masks

Primary position target ialah audited GT trajectory meter, root-relative joint dan root-global pelvis. Primary velocity memakai trailing polynomial endpoint, bukan future/centered smoothing.

**H/T:** Default tujuh sampel degree2, support t−6..t:

```text
u_i = time_i - time_t
q_i ≈ c0 + c1*u_i + c2*u_i²
D0(q)_t = c0
D1(q)_t = c1                # m/s
D2(q)_t = 2*c2             # extension saja, m/s²
```

Untuk waktu irregular gunakan actual timestamps dengan QR/SVD least-squares dan rank/conditioning audit. Untuk grid uniform, endpoint coefficient diuji dengan `savgol_coeffs(7,2,deriv=k,delta=dt,pos=6,use="dot")`; posisi tengah bukan endpoint. [SciPy savgol_coeffs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html).

Enam anchor pertama setiap segmen/window yang reset invalid untuk derivative. Tanpa gap, L32 mempunyai 26 eligible endpoints (6..31). Jangan mengganti dt detik menjadi indeks frame. Jangan resample melintasi gap atau memperlakukan context duplikat sebagai observasi baru. Resampling yang diperlukan membawa source/interpolation provenance, bukan klaim data observasi frekuensi lebih tinggi.

| Mask | Asal | Jalur inference? |
|---|---|---|
| Point/context/frame validity | Finite/padding/keberadaan RPC, timestamp/gap/source chronology | Ya |
| Predicted joint/root validity | Sensor support; jika confidence belum diaudit gunakan broadcast frame validity yang eksplisit | Ya |
| H/M/bin validity | Sensor support dan node/frame policy sama kedua kondisi | Ya |
| Annotation position validity | GT finite/mapping/coordinate audit | Tidak |
| Derivative target validity | Tujuh GT positions valid serta timestamp/gap guard | Tidak |
| Loss/evaluation eligibility | Sensor support ∧ target validity pada besaran terkait | Trainer/evaluator saja |

H finite pada support sensor-valid wajib; NaN/Inf memicu kegagalan numerik, bukan mask baru agar contoh hilang. Target invalid di-select sebelum error dihitung: NaN×0 tetap NaN. Empty group memberi graph-connected zero, count0; bobot dan faktor /2 loss velocity tetap, dengan denominator komponen valid per kelompok. Batch tanpa target valid untuk **seluruh objective aktif** skip optimizer/scheduler dan tidak dihitung sebagai successful update. Pada stage C, GT kinematik kosong tidak menghapus fidelity jika H valid: hanya kontribusi auxiliary menjadi nol, policy dan support sama kedua kondisi.

Untuk tahap C dan matched probes, common_frame_valid = feature_valid AND root_sensor_valid AND all(joint_sensor_valid) AND context_time_valid. Actual memory M pada compressor, auxiliary baseline, dan probe precompression dibatasi latent_valid AND common_frame_valid. Fidelity target serta statistik mu_H/s_H memakai mask sumber yang sama; precompression probe tidak boleh melihat partial frames yang dikeluarkan dari input compressor. Physical position eligibility memerlukan common frame endpoint dan annotation validity; velocity memerlukan seluruh tujuh common frames serta target derivative validity. Mask stage-M readout per-joint tetap estimand upstream yang terpisah. QA cohort memakai kebijakan sensor/target bersama yang dikunci, bukan eksklusi berdasarkan kegagalan condition tertentu.

Root-relative target memerlukan joint dan root valid. Global quantity memakai intersection root/joint; pelvis-relative di-enforce nol dan bukan kemenangan error trivial. Timestamp/support yang sama diperlukan bagi identitas linear `D1(P_relative+r)=D1(P_relative)+D1(r)`.

**R/H:** Coverage reference task mengikuti recipe decoder: ≥0,75 dari 26 eligible endpoints berarti minimal20 setelah guard, bukan 75% dari subset tersisa. Limb comparison memakai intersection support kelompok kiri/kanan. Order memerlukan contiguous support penuh kedua trajectory yang relevan dan quality gate tambahan. Seluruh angka coverage/deadband tetap default yang harus dikunci dari validation sebelum test.

Estimation baseline `D_encoder_sg` memakai **predicted** positions dari common encoder: raw position untuk metric primer, D1 pada trajectory raw untuk velocity. Filtered D0 position terpisah sebagai diagnostic; jangan derivative posisi yang sudah difilter dua kali. Rule baseline tidak membaca GT/annotation masks/params atau memakai future support.

## 7. QA: dua core task dan klaim order bersyarat

Satu task registry dan label recipe bersama menjadi authority generator QA, direct rule, probe, parser dan evaluator. File recipe bukan dibuat otomatis oleh revisi ini. Semua labels berasal dari target offline, bukan caption action IDs.

| Task | Enum answer | Evidence |
|---|---|---|
| `root_speed_trend` — core temporal | speeding_up / slowing_down / no_overall_trend / unknown | Median slope antarpasangan norm(v_root) terhadap detik untuk tren keseluruhan, support/residual/noise gate |
| `relative_limb_motion` — core body relation | faster_left / faster_right / tie / unknown | Perbandingan agregat speed root-relative kelompok joint kiri/kanan, body_part dari pertanyaan |
| `limb_onset_order` — conditional | left_before_right / right_before_left / approximately_simultaneous / unknown | Dua onset limbs terpilih dalam coordinate/root policy audited, crossing/quiet/confirmation/censoring |
| `root_radial_direction` — optional | approaching / receding / stationary / unknown | Root global, origin radar terkalibrasi, radial speed dan range guard |

Posisi joint tidak otomatis membuktikan relasi antarperistiwa. **Tanpa labels `limb_onset_order` yang reliable dan evaluasinya, proposal/hasil hanya boleh mengklaim perubahan temporal dan relasi limbs, bukan pelestarian event order.** Task lama hand–torso tidak otomatis diwariskan; migrasi registry eksplisit bila dipakai sebagai panel berbeda.

Order recipe menyimpan onset time terpisah dari confirmed_at. Default kandidat: quiet awal tiga endpoints, crossing yang bertahan tiga endpoints untuk konfirmasi; early-active, spike, multiple episodes, late confirmation, atau gap ditangani unknown/censored sesuai registry. Deadband/margin dari train/validation, tidak mengarang waktu onset dari label aktivitas. Audit independen pada sampel validation mencakup inspeksi trajectory/reference dan sensitivitas recipe; jika order tidak reliable, hapus klaim order sebelum lock, bukan sesudah hasil test buruk.

Norm acceleration tidak menentukan speeding_up: gerak melingkar dapat ber-speed konstan. `root_speed_trend` menilai tren keseluruhan, bukan monotonisitas setiap frame. `no_overall_trend` berarti slope dalam deadband pada support/fit sah; speed dapat naik lalu turun dan tidak harus konstan. `unknown` tetap untuk support atau kualitas fit yang tidak memadai sesuai recipe dokumen 03. Tie/simultaneous berarti dua besaran reliable dengan selisih kecil, bukan pengganti unknown. Radial stationary bukan seluruh tubuh diam.

### Status target versus jawaban model

| Keadaan | Kontrak scoring |
|---|---|
| Target defined | Satu enum fisik selain unknown; masuk scored panel |
| Target unknown | Recipe sah tetapi ambiguous/insufficient/censored; answer=unknown dengan reason; masuk panel sesuai locked policy |
| Target undefined/excluded | Reference rusak/task tidak applicable; answer=null; dikeluarkan sebelum test dengan reason/count |
| Model unknown pada target defined | Abstention sah secara parser, **tetap salah** pada strict score |
| Model kategori pasti pada target unknown | False certainty; failure pada score dan diagnostic tersendiri |
| JSON invalid atau generation error | Sentinel invalid/error; **tetap salah**, tidak dihapus dari denominator |

Unknown bukan calibrated confidence dan tidak dipakai untuk menyembunyikan prediksi numerik gagal. Numerical evidence undefined dapat menghasilkan target unknown hanya jika recipe/eligibility masih dapat diaudit; schema/reference rusak bukan kelas unknown.

QA row minimal: `qa_id, window_id, subject_id, recording_id, task, body_part?, question_id, question, answer, target_status, reason, observation_interval_s, task_interval_s, label_support, recipe_hash, joint_map_hash, split_hash`. Field legacy `validity` bila dipertahankan harus mapping satu-ke-satu defined↔valid, unknown↔unknown, undefined↔excluded; evaluator menolak kontradiksi.

Satu sensor window boleh mempunyai beberapa QA/paraphrase; dependensinya dicatat. Selected tasks, task intervals, cohort weighting, unknown policy dan metric classes sama bagi kondisi matched. Jangan menghapus QA sulit berdasarkan sensor confidence milik proposed. Counts defined/unknown/undefined serta alasan dilaporkan per task/subject/recording.

## 8. Probe independen: kandungan U bukan kualitas head training

Physical training head bukan probe evaluasi independen. Setelah E/M/C freeze, latih probe baru dari initialization yang matched pada train, pilih validation, lalu test. Tidak memakai auxiliary head C_kin versus head C_base sebagai satu-satunya pembuktian U karena sumber keduanya berbeda.

| Sumber probe | Peran | Yang tidak boleh disimpulkan |
|---|---|---|
| M/H kaya | Reference precompression, joint-time memory256/query | Head stage-M yang memakai residual pose tidak membuktikan H sendiri menyimpan posisi |
| U_base | Keterbacaan sesudah tokenizer baseline | Error tinggi belum membuktikan informasi hilang; bisa probe gagal belajar |
| U_kin | Pelestarian final output pada budget sama | Loss auxiliary rendah belum menjamin generalisasi |
| Z | Diagnostic joint-pooled summary | Bukan sumber penuh U, jadi bukan satu-satunya control compression |
| P=projector(U), optional | Diagnosis interface language embeddings | Tidak wajib menambah scope P0 |

**R/H:** Probe fisik menggunakan query decoder time+joint dengan width256/4 heads, kapasitas output/layer dan parameter count sama untuk M versus U_base/U_kin. M menjadi 32×23 memory256; U menjadi K memory256. Panjang input/time identity berbeda karena kompresi, tetapi query waktu/joint, target dan time support sama. Laporkan compute, sequence lengths, normalization dan param counts; jangan menyebut operasi berbeda panjang identik seluruhnya.

Probe p/v mempunyai query untuk seluruh observed32 frames dan outputs joint/root pada units yang sama. Tidak menerima GT, raw P_enc/root, f_enc, residual bypass, label validity, atau action IDs. Ia boleh memakai sensor times/joint identity yang tersedia bagi semua kondisi. Readout U tidak hanya merekonstruksi mean bin untuk kemudian dibandingkan dengan target per-frame.

Task-probe utama memperoleh jawaban dengan menerapkan recipe deterministik dokumen 03 pada prediksi posisi dan velocity dari probe fisik independen: `M/U → probe fisik → prediksi posisi/velocity → recipe → jawaban task`. Task/body_part menentukan agregasi recipe, bukan masukan tambahan ke probe fisik. Width/layer/head initialization, target posisi/velocity, exposure, tuning dan checkpoint budget matched. Classifier task langsung bukan bagian P0; diagnosis tambahan jika kelak diperlukan dilaporkan terpisah dan tidak mengganti primary setelah melihat test.

Probe dioptimasi sampai convergence check yang wajar pada validation, disertai tiny-overfit dan loss curves. Bila probe underfit atau training tidak stabil, hasil rendah adalah inconclusive. H/M dan U dimuat melalui dtype/support yang sama; memory length adalah treatment, bukan alasan menambahkan informasi GT pada salah satu.

### Diagnostic RQ1

Hitung error/score berpasangan M→U_base dan M→U_kin pada cohort/mask yang sama, lalu interpretasikan bersama fidelity H, task probes, sensor errors dan label quality. Rekonstruksi H mengevaluasi representational fidelity, bukan langsung ground-truth physical correctness.

Jika U_base setara atau lebih baik daripada M pada target terpilih, bottleneck belum didukung: kompresi dapat memberi regularisasi atau membuat target lebih mudah dibaca. Jika M saja tidak reliable, tidak ada reference kuat untuk menuduh kompresi. Jika hanya Z buruk, diagnosis mungkin ada pada joint pooling Z, bukan tokenizer U.

## 9. Frozen LLM, matched alignment, dan grounding

Per kondisi/K, train satu projector scratch independen. Initial states boleh dicloning untuk seed matched, optimizer/checkpoints tetap terpisah. LLM, tokenizer revision, prompt, answer vocabulary/parser, QA IDs, minibatch/exposure, successful update budget, schedule dan validation selection rules sama.

Qwen dan seluruh upstream frozen; tidak ada LoRA atau QA gradient ke compressor. Loss answer-only CE meng-ignore prompt/physical/padding labels dengan −100; internal shifting sekali, EOS/terminator tidak ganda. Frozen LLM forward tetap menyimpan graph agar gradient input mencapai projector; `no_grad` pada seluruh LLM forward akan memutus training projector. [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.6/notes/autograd.html).

Reference generation deterministik `do_sample=false`; answer token budget dipilih validation. Hanya `U`+sensor-derived time/mask encodings dan pertanyaan masuk language path. Target GT/label support/subject/action tidak masuk prompt. Jawaban benar dinilai dengan registry, bukan kemiripan narasi atau LLM-as-judge.

### Baseline dan controls

| Control | Tujuan dan interpretasi |
|---|---|
| Majority train per task/subtype | Prior kelas tanpa sensor; count/imbalance terbuka |
| Text-only Qwen tanpa physical tokens | Prior bahasa; prompt ekuivalen, tidak dianggap learned compressor |
| Direct rule dari predicted states | Keterjawaban sederhana tanpa LLM; recipe/support same dan sensor-only |
| Probe fisik frozen-U → recipe deterministik, tanpa LLM | Apakah kegagalan ada pada representation atau alignment; jawaban task diturunkan dari prediksi posisi/velocity |
| Donor recording nyata, same activity dengan reference answers berbeda | Same question/compatible time support; jawaban dibandingkan dengan **label window masing-masing**, bukan label penerima saja |
| Donor same reference answer | Invariance control; bukan bukti model mengikuti perubahan gerak |
| Sensor corruption/bin shuffle | Sensitivitas/OOD diagnostic; bukan physical counterfactual |
| Oracle GT, jika diperlukan | Upper-reference terpisah; bukan deployment dan bukan control primer |

**R:** Panel grounding P0 memilih pasangan recording berbeda **dalam aktivitas yang sama**, dengan kedua reference targets defined dan answers berbeda. Pertanyaan/task/body_part sama; coordinate policy, observation duration/task interval, valid derivative support dan chronology harus compatible. Window asal A dan donor B masing-masing dievaluasi terhadap y_A dan y_B. Action ID tidak masuk prompt. Donor IDs/eligibility dan selection policy dikunci sebelum outcome test; pemilihan same-label tidak menjadi bukti perubahan. Jika cohort berbeda jawaban tidak tersedia, laporkan batas grounding/coverage dan jangan menggantinya dengan arbitrary latent corruption.

Metrik grounding, dengan denominator pasangan reference-eligible yang fixed:

```text
PairBothCorrect = mean(pred_A == y_A AND pred_B == y_B)
CorrectChange = mean((pred_A, pred_B) == (y_A, y_B) AND pred_A != pred_B)
RawChangeRate = mean(parsed predictions valid AND pred_A != pred_B)
```

Untuk target kategorikal berbeda, CorrectChange setara secara matematis dengan PairBothCorrect: keduanya dilaporkan agar arti perubahan benar jelas, **bukan dua bukti independen**. RawChangeRate tinggi sendiri tidak cukup. Parse invalid/error pada salah satu anggota membuat pair failure, bukan exclusion. Report candidate-pair count, reference-eligible count/coverage, pair parse coverage, per-task/transisi kelas dan missing/unknown reasons. Same-label pairs dilaporkan sebagai invariance panel terpisah. `ConditionalCorrectChange` opsional membagi jumlah pasangan yang kedua jawabannya benar dengan jumlah pasangan yang jawaban A benar; denominator nol menghasilkan undefined. Conditioning/coverage wajib dicatat dan metrik ini tidak menggantikan strict pair score.

GT tidak menjadi input deployment. Mengubah latent arbitrary bukan intervensi gaya/massa dan bukan bukti causal understanding. Reversal opsional harus rebuild timestamps/context/masks dan labels dari trajectory yang relevan; membalik cached latents hanya feature corruption. Tidak wajib setiap perturbasi menurunkan semua task.

**R:** Keunggulan QA yang beriringan dengan probe improvement mendukung manfaat downstream pada panel ini, tetapi tidak membuktikan mediasi kausal tunggal: fidelity, optimisasi projector dan representasi lain dapat ikut berubah. Klaim lebih kuat perlu kontrol tambahan, bukan narasi sebab-akibat dari korelasi skor.

## 10. Metrics, estimands, dan ketidakpastian

### Physical/readability metrics

| Besaran | Metrik | Unit/support |
|---|---|---|
| Joint position relatif | MPJPE tanpa Procrustes; maksimal21 nontrivial joints | mm; pelvis constrained-zero dikeluarkan |
| Root position dan global joints | Root trajectory error; global MPJPE22 audited joints | mm; root tetap terpisah |
| Velocity | Mean vector error; speed norm MAE, joint-relative/root-global terpisah | m/s; trailing-seven support yang sama |
| H reconstruction | Train-normalized component MSE + raw channel error | Bukan error meter; all-valid node/time counts |
| Core task probe/LLM | Accuracy, macro-F1, confusion dan per-class support | Registry/QA cohort locked |
| Onset order bila dipilih | Order score; onset MAE/detection/missing/censor counts | Detik dan kategori; order claim conditional |
| Resource | Counts, bytes, measured latency/VRAM/GPU-hours | Units dan stage berbeda, bukan estimate=measurement |

Continuous predictions NaN/Inf pada eligible support dicatat sebagai numerical failures/missing predictions. Error pada finite common support disertai failure rate seluruh cohort; jangan menghapus kegagalan agar error membaik. Global/root dan articulation tetap terpisah. Bone-length/velocity-consistency, angle/acceleration/phase, PA-MPJPE atau mesh error hanya diagnostic/perluasan dengan target sesuai.

### Estimand primer

K16 primer; physical retention dan language utility merupakan endpoint berbeda:

```text
Delta_position = error(U_base) - error(U_kin)         # positif: proposed lebih baik
Delta_velocity = error(U_base) - error(U_kin)
Delta_probe = score(recipe(probe_U_kin)) - score(recipe(probe_U_base))
Delta_LLM = score(LLM_U_kin) - score(LLM_U_base)
RQ1_gap = error(U_base) - error(M)                    # interpret dengan probe limitations
```

Pada `Delta_probe`, probe_U_kin/probe_U_base menyatakan prediksi posisi/velocity dari probe fisik independen, recipe mengubahnya menjadi jawaban task, dan score memakai agregasi QA di bawah. Delta ini bukan skor classifier task langsung. Delta fisik tidak digabung sembarang dengan mm dan m/s. Report tiap kelompok/root/joint dengan common support/counts. RQ1 task gap memakai score(recipe(probe_M))−score(recipe(probe_U_base)); tidak memaksa semua gap positif.

QA score reference:

```text
F1_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
macro_F1_task = mean(F1_c for locked metric_classes)
score = mean(macro_F1_task for selected_tasks)
```

Metric classes default empat enum termasuk unknown, `zero_division=0`; registry/support/kelas tanpa target dicatat. Kelas tanpa target memberi F1 nol dan dapat membatasi ceiling macro-F1; batas ini dilaporkan tanpa mengganti denominator sesudah test. Bila policy lain dipilih validation, kunci sebelum test dan tetap laporkan semua enum. Jangan mengubah kelas/task denominator setelah melihat hasil.

Invalid prediction sentinel menambah FN true class dan tidak hilang dari score. Confusion memiliki kolom invalid. Missing generation/error juga failure. Parser menolak duplicate keys, wrong task/enum, extra fields/prose dan malformed JSON tanpa memperbaiki jawaban. Parse coverage, accuracy semua QA, abstention defined, false certainty unknown, dan conditional accuracy dilaporkan; conditional accuracy bukan score primer.

### Grouped uncertainty

**R/T:** Subjek adalah kelompok generalisasi; recording/windows/QA saling bergantung. Laporkan N_subject, N_recording, N_unique_window, N_QA, class support dan overlap. Resampling frame/QA seolah independen tidak sah.

Paired subject-level atau hierarchical subject→recording bootstrap dipilih menurut estimand dan audit cohort. Seluruh paired predictions, tasks dan windows dipertahankan bersama; hitung ulang metrics dari weighted confusion pada setiap draw. Default estimand row-weighted mengikuti locked QA sampling; subject-macro sensitivity dipisah, bukan dicampur sebagai estimand yang sama. [Metode hierarchical bootstrap](https://arxiv.org/abs/2007.07797).

Simpan unsupported-task/invalid-resample counts; tidak mengurangi task set per draw. Interval level/method, jumlah draws dan stability check ditetapkan development, bukan angka yang menjamin precision. Jika cluster terlalu sedikit, laporkan descriptive per-subject deltas dan batas uncertainty; jangan memberikan kepastian semu. Donor panel yang memakai donor berulang atau pasangan lintas subject mempunyai dependensi tambahan: laporkan IDs/counts serta descriptive pair metrics; interval QA subject-only tidak otomatis sah bagi panel donor tanpa grouping yang menangani kedua anggota.

Satu seed P0 sah sebagai **exploratory PoC**, bukan bukti robust terhadap variasi training. Bootstrap conditional pada fitted checkpoints tidak mengukur seed variance. Seeds tambahan matched bila mengklaim robustness; jangan menganggap tiga seed sebagai tiga cohort independen.

Tolerance efek praktis ditetapkan train/validation berdasarkan label stability, task utility, metric scales dan biaya, tanpa success threshold buatan. “Tidak signifikan” bukan bukti setara. Klaim tidak ada manfaat praktis memerlukan uncertainty cukup sempit relatif tolerance; interval lebar berarti inconclusive. Banyak secondary budgets/task analyses tetap secondary atau exploratory.

## 11. Acceptance tests sebelum training penuh

Ini **spesifikasi test yang harus diimplementasikan**, bukan test yang telah dijalankan. Reuse satu script assert-based CPU/data/GPU; tidak perlu framework baru. Toleransi kandidat fp64 polynomial1e−8, physical fp32≈1e−4, causal/padding≈1e−5 pada fixture berskala wajar, lalu pin sesuai precision/backend. Jangan melonggarkan toleransi untuk menutupi unit/leakage.

| Test | Fixture / assertion |
|---|---|
| DATA-SPLIT | Subject/recording/source union disjoint; key ordering, duplicates/gap/context rejection benar |
| SENSOR-ONLY | U/logits tetap ketika GT/params/bbox/action/annotation masks diubah atau aksesnya diblok |
| COORD | Relative+root=global; pelvis zero; transform inverse dan origin konsisten; trans bukan auto pelvis |
| DERIVATIVE | Constant/linear/quadratic, uniform dan irregular timestamps; D1 units detik; first6 invalid |
| SUPPORT | Missing root/joint, duplicate time, gap; mask conjunction benar; finite zero tidak dianggap valid diam |
| CAUSAL-TRUNK | Prefix E/H sama ketika suffix radar sesudah t berubah; eval/train normalization diuji |
| WINDOW-READOUT | Auxiliary/probes U diperlakukan full-observed-window retrospective; tidak menuntut causal invariance per-timestep yang salah |
| CACHE | Same context/reset/time/window through train/eval/cache/infer; incomplete/foreign hashes rejected |
| BIN-MASK | K8/16/32 chronology/provenance benar; query tidak question-dependent; empty bin zero/mask |
| H-FIDELITY | Decoder queries seluruh32×23, output128; H detached; full-H target tidak terganti Z |
| NO-BYPASS | U-readout dilarang menerima GT/predicted-state skips/f_enc; input signatures dan access guards diuji |
| AUX-ROUTE | Base auxiliary grad mencapai head saja; kin auxiliary mencapai head+C; H/motion/encoder tidak berubah |
| AUX-CLIP | Pada C_base finite, mengubah auxiliary loss/gradien tidak mengubah gradien/update compressor sesudah clipping terpisah |
| MEMORY-PE | M_raw cache immutable; M_attn PE waktu tepat sekali dan identik pada compressor/auxiliary baseline/probe; U tidak diberi PE memory ulang |
| PAIRED-INIT | q0/joint embeddings/heads/compressor memiliki initializer, decay policy, cloned state dan checksum yang tercatat |
| COMMON-REC-GRAD | Fidelity menghasilkan finite nonzero C/decoder gradient pada kedua kondisi |
| PROBE-MATCH | Head counts/capacity/labels/time/support sama; fresh heads tanpa reuse auxiliary weights; jawaban task berasal dari prediksi p/v melalui recipe, tanpa query task/body_part ke probe fisik |
| LOSS-MASK | Select invalid targets sebelum arithmetic; count0 graph-zero; semua objective invalid skip update; H valid tetap melatih fidelity walau auxiliary GT kosong |
| FREEZE | Weight hashes fixed, frozen IDs di luar optimizer; trainable update benar pada setiap stage |
| LLM-INPUT-GRAD | Qwen frozen tetap memberi gradient finite ke projector; upstream tidak mendapat QA grad |
| LLM-LOSS | Answer-only labels, CE shift sekali, EOS/padding benar |
| LLM-GENERATE | Teacher-forced first-answer raw logits konsisten dengan deterministic generation; batch/padding positions benar |
| TREND-LABEL | Fixture 26 endpoint naik–turun dokumen 03 §9.1 menghasilkan slope0/MAD0 dan no_overall_trend; speed konstan juga no_overall_trend; support kurang unknown |
| EVENT | Jika selected: quiet/on/spike/rearm/gap/early-active/late confirmation; onset≠confirmed_at |
| DONOR-GROUNDING | Same activity, different recording/defined answers, same question/compatible time support; both-correct/correct-change denominator fixed; same-label dipisah sebagai invariance |
| PARSE-SCORE | Invalid JSON/extra field ditolak; satu correct+satu invalid kelas sama menghasilkan **F1 kelas target=2/3**; macro-F1 mengikuti registry (empat kelas, tiga unsupported dengan F1 nol → macro-F1=1/6) |
| RESUME | Checkpoint+RNG/sampler/optimizer/scaler restored; foreign contract/cache/condition rejected |
| LOG-REPORT | Fixture artefak §12.2 lengkap lolos; missing history/prediction/ID/hash/count dan completion palsu ditolak; metric sum/count dihitung ulang; failed/incomplete tetap menyimpan reason/status |

Pure-PyTorch dan `.venv`/uv wajib untuk Python aktual; dependency native baru tidak dipasang tanpa persetujuan. Smoke model nyata mencakup Qwen backward-input, bukan dummy saja. Tiny-overfit train-only pada subset kecil memeriksa pipeline bisa belajar, bukan generalisasi.

## 12. Gate penelitian dan interpretasi hasil

| Gate | Deliverable | Jika gagal |
|---|---|---|
| G0 data/targets | Schema/time/coordinates/joint/subject split, derivative audit | Hentikan scientific run, perbaiki audit; belum negative result |
| G1 implementation | Tests/smoke/tiny-overfit/freeze/gradient routes | Diagnosis implementasi atau numerik, bukan hipotesis |
| G2 common upstream | E/M checkpoints dan reliable physical targets/support | Batasi task atau perbaiki train/validation sebelum lock |
| G3 compression | C_base/C_kin full-H fidelity, U probes dan resource profile | Diagnosis fidelity/probe/optimisasi; jangan sengaja melemahkan baseline |
| G4 task/grounding | Dua core tasks, order optional reliable, donor cohort | Klaim mengikuti panel yang sah; order tidak dipaksakan |
| G5 matched alignment | Dua projector, same IDs/updates/parser/model revisions | Run partial/crash bukan final matched |
| G6 locked test | Seluruh endpoint/failures/grouped uncertainty/resource | Jangan retune dari test; laporkan seluruh selected outcomes |
| G7 reproduction | Hash bundle, satu prediction dan tabel dapat dihitung ulang | Catat keterbatasan reproducibility, tidak menghapus hasil historis |

Engineering gate tidak mensyaratkan hipotesis positif. U_base yang sudah cukup baik tetap hasil sah; task dipilih berdasarkan reliabilitas reference, bukan karena proposed menang.

| Pola hasil sah | Interpretasi |
|---|---|
| M baik, U_base memburuk, U_kin lebih baik, QA grounded ikut membaik | Dukungan pelestarian output-side dan manfaat downstream pada setup/budget ini |
| U_kin physical/probe lebih baik, QA tidak membaik | Metode representasi didukung; manfaat frozen-LLM belum didukung |
| U_base setara M dan U_kin tidak memberi keuntungan | Dugaan bottleneck belum didukung pada target/budget ini; jangan klaim masalah kompresi universal |
| U_base memburuk, U_kin tidak membantu | Masalah terukur tetapi metode usulan belum menyelesaikannya |
| Training head H baik dengan residual skip, probe M buruk | Belum ada bukti informasi terbaca dari H; jangan menuduh U/LLM |
| Probe U baik, QA lemah | Batas alignment/interface atau kemampuan model bahasa; bukan otomatis kegagalan tokenizer |
| QA baik, text-only setara atau donor tidak diikuti | Grounding sensor belum didukung; shortcut/prior masih mungkin |
| Tidak ada gain dengan interval lebar | Inconclusive, bukan equivalence atau hipotesis pasti salah |
| Label/GT alignment buruk, NaN, freeze salah, optimizer gagal | Engineering/data failure; bukan negative result ilmiah |
| Protokol reliable, precision memadai, baseline kuat, proposed tidak unggul | Negative result bermakna untuk metode/task/split/budget yang diuji |

Hasil positif tidak otomatis menetapkan kebaruan/SOTA. Hasil negatif yang valid tetap dapat menjelaskan batas pendekatan, tetapi bukan bukti seluruh AI motion/tokenization mustahil. Jangan mengganti objective sesudah test demi narasi keberhasilan.

### 12.1 Kesiapan eksperimen pertama dan development lintas perangkat

**Eksperimen pertama/P0** berarti studi minimum K16 pada dua core tasks, dengan satu encoder/backbone common, paired C_base/C_kin, probes independen, dua projector serta controls yang sudah ditetapkan pada §4. **Tahap E** adalah training encoder pertama di dalam P0, bukan keseluruhan studi. P0 satu seed bersifat exploratory. Planning cukup untuk memulai development; kesiapan scientific run baru dinyatakan setelah gates terkait lulus pada data/perangkat nyata.

Development dapat dilakukan sebelum dataset tersedia: implementasikan kontrak tensor, reader yang gagal jelas jika path/schema tidak sesuai, recipe, models, train/evaluate loops, logging, resume dan satu self-check sintetis. Fixture diberi `data_kind=synthetic`; input dan targetnya tidak menjadi fallback otomatis bagi run `data_kind=m4human`. Hasil fixture memeriksa implementasi, bukan reliabilitas label atau performa radar nyata. Audit schema/unit/time/joint map tetap berstatus belum diverifikasi sampai perangkat dataset menjalankan G0.

Handoff development membawa kode+commit/dirty diff, dependency/runtime specification, config template, fixture/self-check, serta petunjuk urutan audit→target/split→smoke/tiny-overfit nyata→profil→training. Path diisi melalui config pada perangkat dataset. Jalankan ulang self-check di perangkat tujuan, lalu lengkapi null audit fields dan gates sebelum full training. Smoke Qwen hanya ditandai lulus jika memakai model nyata; fixture dummy tidak menutup gate Qwen backward/generation. Task panel, arsitektur dan scientific comparison tetap sesuai kontrak P0.

### 12.2 Kontrak log minimum untuk analisis pascarun

Tujuannya ialah memungkinkan peneliti menghitung ulang hasil, menemukan tahap yang membatasi performa, dan memilih perbaikan terarah. **Log lengkap tidak menjamin penyebab tunggal dapat ditemukan atau performa pasti meningkat.** Diagnosis dapat tetap inconclusive dan memerlukan inspeksi target/source atau pemeriksaan tambahan pada train/validation. Kontrak ini berlaku sejak smoke/pilot dan tahap E, bukan baru setelah seluruh P0 selesai; field yang tidak berlaku bagi suatu stage memakai status `not_applicable` dengan alasan.

Gunakan folder run pada §14, dengan `run_id`, `parent_run_id` bila iterasi/resume, `stage`, `condition`, `K`, `seed`, dan `data_kind`. JSON/JSONL serta NumPy yang sudah direncanakan cukup; tidak memerlukan layanan tracking baru. Artefak minimum:

| Artefak dalam run | Isi dan fungsi |
|---|---|
| `config_snapshot.yaml`, `lineage.json` | Config resolved, source/split/recipe/cache/model hashes, normalizers, coordinate/joint maps, code commit+dirty diff, versions, device, precision/backend, seed dan trainable map; membedakan perubahan model dari perubahan data |
| `history.jsonl` | Ringkasan training per epoch dan setiap validation: epoch/update, samples/windows/QA exposure, loss per term tanpa bobot serta total berbobot, valid counts/reduction policy, learning rate, gradient norm per grup sebelum/sesudah clipping, scaler/skipped updates, elapsed time, throughput dan RAM/VRAM aktual |
| `events.jsonl` | Start/resume, validation, checkpoint selection dengan metric/tie-break, stop/complete, dan error dengan stage/update/source IDs serta traceback/reason; error/overflow/skip dicatat segera |
| `predictions_val.jsonl`, `predictions_test.jsonl` | Records pada cohort fixed untuk checkpoint yang dipilih: IDs/support/condition, metric fisik per-window/per-joint dengan error sum/count, jawaban task dari recipe, reference IDs, parse/failure status; tahap L juga menyimpan raw generation IDs/text. Test hanya dibuat saat evaluasi final yang telah dikunci |
| `diagnostic_samples.npz`, `diagnostic_samples.json` | Panel kecil tetap train/validation, dipilih sebelum training dan shared antar kondisi matched: p/r/v predicted dan reference, time/masks/support/IDs. Simpan untuk checkpoint terpilih guna inspeksi trajectory/alignment; sampling policy/count dipin, tanpa dump seluruh activation/RPC tiap step |
| `metrics.json`, `report.md` | Metrik per task/class/joint/subject/recording, confusion, valid/unknown/excluded/parse/failure counts, matched deltas+uncertainty sesuai tahap, loss curves, biaya dan diagnosis; setiap hasil menyebut checkpoint/cohort/unit/denominator |
| Checkpoint best/last dan metadata | State serta selection/provenance sesuai stage; last menyimpan resume state yang telah ditentukan. Paths/hash dicatat, tanpa menduplikasi frozen LLM |

History tidak hanya menyimpan satu total loss: E memisahkan pose/root; M memisahkan empat term p/v; C memisahkan full-H fidelity dan empat term auxiliary; probes memisahkan p/v dan skor task hasil recipe; L memisahkan answer-only CE dari generated validation scores. Scalar telemetry tambahan per-update boleh diringkas pada interval yang dipin dalam config; epoch/validation summaries dan kejadian gagal tetap wajib. Record numerator/count untuk metrik serta kebijakan agregasi loss; rata-rata microbatch tidak dipresentasikan sebagai rata-rata semua komponen tanpa weighting yang sesuai. Clipping per grup mengikuti tahap C; telemetry per grup adalah field terpisah jika tersedia. Scalar maximum norm helper tidak dilabel sebagai setiap norm grup.

Evaluasi training menggunakan panel train kecil tetap dengan eval mode dan recipe yang sama dengan validation untuk membantu membedakan underfit dan overfit; tidak memakai perbandingan langsung training loss yang memakai dropout/augmentasi dengan validation metric. Panel ini shared antar kondisi, tidak dipakai memilih sampel yang mudah, dan dicatat sebagai diagnostik. Detail prediction/diagnostic arrays disimpan pada checkpoint terpilih, bukan seluruh checkpoint setiap epoch. Nilai non-finite serialisasi sebagai `null` dengan failure flag/reason; tidak menulis NaN/Infinity sebagai metrik sah. Missing output pada cohort eligible tetap failure row.

Pada setiap validation/checkpoint dan sebelum exit normal, flush log dan simpan resume state pada batas yang didukung. Exception menghasilkan status `failed` bila handler masih dapat berjalan; proses yang terputus tanpa completion marker dianggap `incomplete`, bukan run selesai. Simpan artefak partial untuk diagnosis. Run dinyatakan **siap dianalisis** hanya setelah validator memeriksa file/schema/IDs/hashes/counts, checkpoint-selection lineage, dan perhitungan ulang metrik dari records+reference manifests. Untuk paired comparison, kedua run harus memiliki cohort/support/exposure yang sepadan; run tunggal yang lengkap hanya menutup analisis stage-nya sendiri. Jika artefak minimum hilang, report menyatakan diagnosis terbatas dan tidak menjanjikan analisis lengkap.

### 12.3 Diagnosis dan perbaikan setelah run pertama

Gunakan bukti per-stage berikut bersama tabel interpretasi §12. Gejala memberi kandidat penyebab, bukan kepastian kausal:

| Gejala pada train/validation | Bukti yang diperiksa | Perbaikan terarah dalam scope P0 |
|---|---|---|
| Root/global error besar walau relative pose baik | E pose/root losses, per-subject/joint counts, unit/transform/origin dan diagnostic trajectories | Perbaiki adapter/unit/map bila salah; jika sah, periksa optimization encoder pada train/val |
| Loss tidak turun atau prediction hampir tetap | Tiny-overfit, actual successful updates/LR, grad per grup, target scales/counts, freeze/optimizer IDs | Perbaiki loss/mask/autograd/resume bug; uji LR/jadwal dalam budget yang ditetapkan |
| Train baik tetapi validation buruk | Same-metric train panel versus validation, class/subject coverage, split/source support | Periksa leakage/distribusi dan pilih checkpoint sesuai rule; regularisasi atau jadwal hanya melalui train/val |
| M probe buruk meski full-H residual head baik | Fresh M probe p/v, convergence, encoder quality, residual bypass dan target reliability | Diagnosis upstream atau probe; jangan menyimpulkan informasi hilang akibat kompresi |
| M baik, U buruk | Paired M/U metrics, full-H fidelity, per-joint/time errors, mask/bin/PE, C gradient routes | Perbaiki packing/recipe bug; jika valid, uji optimization atau lambda_aux bersama pada K16 |
| U probe baik, QA buruk | L answer-only CE, generated validation, parse/confusion, prefix/generation equivalence dan projector gradient | Perbaiki alignment/prefix/parser implementation atau optimization projector; tetap gunakan frozen Qwen dan U |
| QA tinggi tetapi text-only setara atau donor tidak diikuti | Class prior, fixed donor labels/support, both-correct/coverage dan per-task raw responses | Audit shortcut/grounding; batasi kesimpulan jika bukti sensor belum cukup |
| OOM, overflow atau run lambat | Peak allocated/reserved/RSS, I/O, length, scaler/skips dan precision smoke | Profil lalu sesuaikan microbatch/accumulation atau native checkpointing secara matched; ulang numeric gate |

Setelah run, `report.md` memuat status gate dan bottleneck yang didukung, bukti+alternatif penjelasan, paling banyak satu perubahan utama untuk iterasi berikutnya, metric validation yang diharapkan berubah, risiko regresi, dan run/cohort yang akan dibandingkan. Jika tidak ada bukti yang cukup untuk perubahan, keputusan yang sah adalah inspeksi terbatas atau berhenti; jangan memperbesar model secara otomatis. Catat budget putaran/GPU-hours sebelum iterasi perbaikan dimulai. Iterasi berhenti ketika budget habis, tidak ada kandidat berbasis bukti, atau tujuan/gates PoC telah tercapai; kenaikan skor bukan syarat untuk mengakui hasil negatif yang valid.

Perbaikan model/hyperparameter memakai **train/validation**, dicatat melalui `parent_run_id` dan config diff, dan dibandingkan pada cohort/support yang sama. Untuk C_base/C_kin, gunakan tuning/exposure/initialization policy sepadan; perubahan common encoder/backbone menghasilkan cache baru bersama dan rerun dependent stages. Perbaikan label/map/unit memperbarui recipe/hash/cohort secara eksplisit sebelum test lock. Jangan membandingkan hasil dari recipe/cohort berbeda seolah hanya model yang berubah. Trial tidak menambah task, backbone alternatif, LLM atau budget token di luar P0. Hasil “lebih baik” dinyatakan dari validation effects dan regression checks yang dipin, lalu dikonfirmasi pada locked test sesuai batas eksploratif satu seed.

Setelah test dibuka, temuan boleh dianalisis untuk laporan tetapi test tidak menjadi dasar tuning model, threshold, label recipe atau selection. Perubahan yang terinspirasi hasil test berstatus exploratory pada cohort itu; klaim konfirmatori baru memerlukan held-out data baru yang belum dipakai memilih perubahan. Planning ini tidak menjanjikan bahwa test yang sama dapat dipakai berulang untuk memperbaiki dan membuktikan generalisasi.

## 13. Resource: aritmetika versus pengukuran

Target perangkat eksperimen: satu RTX3060 12GB dan RAM16GB. Semua kandidat batch/dtype/angka memori di bawah **H/T**, bukan hasil benchmark atau jaminan muat.

| Tensor/bagian | Hitungan referensi | Tidak termasuk |
|---|---|---|
| H satu window fp16 | 32×23×128×2 = 188.416 byte | Activations trunk/gradients/container |
| M satu window fp16 | 32×23×256×2 = 376.832 byte | H dapat disimpan lalu M dibentuk on-demand; jangan wajib duplikasi disk |
| Z satu window fp16 | 32×256×2 = 16.384 byte | Metadata/time/masks |
| U K8/K16/K32 fp16 | 4.096 / 8.192 / 16.384 byte | Projector activations/LLM input tensors |
| Full-H queries width256 fp16 | 32×23×256×2 = 376.832 byte | Attention/backward dan output128 |
| Ilustrasi 30.000 H windows | 188.416×30.000 = 5.652.480.000 byte ≈5,65GB desimal | Overlap, GT/IDs/masks, M/Z, files/container; window count belum diaudit |
| Projector | 2×256+256×512+512+512×1536+1536 = 920.064 parameter | Grad/master/Adam states |
| Frozen LLM weights | N_actual×bytes(dtype); ilustrasi1,54B×2≈3,08GB | Input-grad activations/KV/workspace/allocator |

Compressor, full-H decoder dan auxiliary/probe param counts diukur runtime dari instansiasi, bukan disimpulkan dari nama. Laporkan total/trainable/frozen counts per stage/condition/K. Full-H query attention menambah beban dibanding Z-only; profile tiny workbatch sebelum final. Panjang K mengurangi token bahasa tetapi H/M compute upstream tetap ada.

Mulai streaming/memmap, workers0, microbatch1. Tuning accumulation/precision bersama dilakukan setelah profile; effective batch=microbatch×accumulation pada satu GPU. E/M/C/probe/Qwen dijalankan bergiliran; cache frozen memisahkan training mahal. RAM16GB tidak memuat seluruh LMDB/H cache sekaligus.

Ukur GPU/driver, resolved precision/backend, RSS, allocated/reserved/peak VRAM, I/O wait, median/p95 successful update time, examples/s, validation/extraction/checkpoint overhead, cache bytes dan GPU-hours. Timing CUDA memakai synchronization/events. Duration estimasi memakai measured time **per successful optimizer update termasuk microsteps**, bukan ukuran download50GB.

OOM: turunkan microbatch/naik accumulation dahulu. Mengubah N/L/width/K mengganti kontrak/estimand terkait dan mengulang gate; tidak hanya mempermudah proposed. Skipped scaler update dicatat, bukan successful matched update. NaN/Inf, leakage, salah split/hash, frozen weight berubah atau gradient putus menghentikan run.

Latency sensor→E→M→C→projector→generation dipisahkan dari cold-load, cached-only inference, span observasi dan jumlah answer tokens. Jangan menyebut real-time dari latency cache atau weight memory saja.

## 14. Config, API, dan artefak reproduksi

Contoh berikut blueprint, bukan config yang sudah dibuat:

```yaml
contract_version: m4human_kinetok_v3
status: planned
seed: 42
data:
  root: /dataset
  schema_hash: null
  joint_map_hash: null
  split_hash: null
  timestamp_policy: audit_required
  input_channels: [x, y, z, intensity]
encoder:
  reference: M4HumanSetEncoderV2
  initialization: scratch
  causal_context: 4
  points: 512
  joints_after_audit: 22
  feature_dim: 256
motion:
  reference: CausalDSTformerLiteV1
  initialization: scratch
  length: 32
  spatial_nodes: 23
  width: 128
  stages: 4
  frozen_before_comparison: true
compression:
  reference: KinematicTokenLearnerV1
  width: 256
  heads: 4
  primary_budget: 16
  predefined_secondary_budgets: [8, 32]
  secondary_budget_status: feasibility_not_measured
  tokens_per_bin: 1
  fidelity_target: full_frozen_H
  fidelity_query: time_joint
  fidelity_query_width: 256
  fidelity_output_width: 128
  fidelity_normalizer_hash: null
  conditions: [C_base, C_kin]
  auxiliary_source: {C_base: M, C_kin: U}
  auxiliary_targets: [position_joint, position_root, velocity_joint, velocity_root]
  auxiliary_raw_state_bypass: false
  auxiliary_readout_scope: retrospective_observed_window
  lambda_aux: 1.0                     # H default; final locked on validation
  resolved_training_budget_hash: null
derivative:
  window: 7
  degree: 2
  evaluation: trailing_endpoint
  warmup_invalid: 6
  actual_timestamps_if_available: true
language:
  model: Qwen/Qwen2.5-1.5B-Instruct
  revision: null
  fully_frozen: true
  projector: LN256_Linear512_GELU_LinearConfigHidden
  train_projector_only: true
  no_lora: true
evaluation:
  core_tasks: [root_speed_trend, relative_limb_motion]
  event_order_claim: false
  conditional_order_task: limb_onset_order
  matched_probe_sources: [M, U_base, U_kin]
  secondary_probe_source: Z
  recipe_hash: null
  test_contract_hash: null
  practical_effect_tolerances: null
  grouped_uncertainty_policy: null
precision:
  reference: fp16_amp_with_grad_scaler
  alternative: bf16_only_if_supported_and_tested
  resolved: null
```

Mandatory nulls diselesaikan pada audit/development sebelum final run. Unknown model/task/enum tidak fallback diam-diam. Jika recipe IDs lama dipakai, cocokkan isi/hash dengan kontrak baru; version string saja bukan bukti kompatibilitas.

API rencana minimum:

```text
read_sensor_context(keys) -> RPC, sensor_masks, timestamps, source_support
read_targets_offline(keys) -> audited GT, annotation_masks
encoder(...) -> predicted_relative_pose, predicted_root, f_enc
motion(...) -> H, Z
build_memory(H) -> M
compressor(M, sensor_masks, times, K) -> U, token_mask, bin_provenance
reconstruct_H(U, token_mask, time_joint_queries) -> H_hat
aux_readout(source=M_or_U, query_times_joints) -> positions, velocities
fit_independent_probe(frozen_source, training_targets) -> probe_checkpoint
projector(U) -> physical embeddings
reason(bundle, sensor_window, question) -> raw_structured_answer
score(test_contract, QA_reference, raw_predictions) -> metrics, failures, coverage
paired_grouped_effect(records, group_manifest, policy) -> deltas, uncertainty
```

Reuse satu reader/recipe/parser/evaluator dari implementasi yang kelak dibuat; tidak perlu library evaluasi baru atau duplikasi formulas. Source targets offline dan API inference dipisah agar GT tidak menjadi argumen reason. Commands tidak ditulis seolah scripts baru sudah tersedia.

Run aktual menambah folder laporan `docs/report_training/RUN_M4HUMAN_<timestamp>_<stage>/` dan INDEX setelah run tercatat; rencana ini tidak dimasukkan sebagai training selesai. Jangan menghapus laporan historis.

Bundle minimum: source/data/joint/time/coordinate versions; split/window/QA manifests; normalizers/label/bins recipe; E/M/C/probe/projector checkpoints; environment/model/tokenizer revisions; config+dirty diff; trainable/frozen/gradient-route records; checkpoint-selection log; test contract; raw predictions/errors; grouped results; measured resource. Field `measured/estimated/not_run` dibedakan. Artefak cache menyebut complete/incomplete dan conditioning IDs.

Reproduction checks sesudah implementasi:

1. Satu sensor-only sample memberi same source/time hashes dan tensors dalam toleransi, tanpa akses params.
2. M/H, full-H fidelity target, C_base/C_kin dan stage freeze/routes cocok dengan bundle.
3. U yang diprobe identik dengan U yang masuk projector; no skip route.
4. Satu QA generation dan seluruh scoring table dihitung ulang dari raw records, termasuk invalid/error.
5. Resume satu update memakai RNG/sampler/optimizer/scaler yang tersimpan; catat batas determinisme backend/hardware.
6. Validator artefak §12.2 menolak missing log/IDs/hash/counts atau completion marker palsu; run terputus tetap incomplete. Recompute metrik stage dari records+reference manifests cocok dengan metrics.json dalam toleransi numerik yang dipin.

## 15. Checklist integrasi final

- [ ] Kontrak seluruh dokumen ialah `m4human_kinetok_v3`; nama model lokal bukan release opensource.
- [ ] Pertanyaan RQ1/RQ2/RQ3/RQ4 dan batas kinematika/frozen LLM jelas; tidak mengasumsikan RadarLLM gagal atau output-side loss/token-to-language belum mempunyai pendahulu.
- [ ] /dataset, XYZ/intensity, timestamp, calibration, 22 joint/pelvis, recording/split diaudit.
- [ ] Common encoder/motion frozen; H/M kaya dan Z diagnostic mempunyai support sama.
- [ ] Fidelity primer kedua kondisi merekonstruksi full H dengan query time+joint/output128, bukan Z-only.
- [ ] Auxiliary M pada C_base versus U pada C_kin memakai head/labels/scales sama; perbedaan gradient placement dicatat.
- [ ] U K16×256 menjadi input nyata projector; K8/32 secondary pradefinisi atau klaim multi-budget dibatasi.
- [ ] Probe M/U independen dan matched; residual state/GT/feature bypass tidak ada.
- [ ] Dua core tasks reliable; tanpa audited order task tidak ada event-order claim.
- [ ] Dua projector-only matched, Qwen/upstream frozen, no LoRA/no QA gradient upstream.
- [ ] Test lock, invalid/unknown coverage, common support, practical tolerance dan clustered uncertainty dicatat.
- [ ] Donor same-activity/beda-answer mempunyai both-correct/correct-change serta coverage; same-label adalah invariance; shuffle/OOD bukan physical counterfactual.
- [ ] Negative result dibedakan dari target/engineering/probe failure dan precision rendah.
- [ ] Hasil/resource yang belum diukur tetap rencana, tanpa success number atau feasibility guarantee.
- [ ] Development sintetis dan handoff perangkat dataset mengikuti §12.1; audit serta smoke nyata belum digantikan fixture.
- [ ] Kontrak log §12.2 sudah diimplementasikan sejak pilot/E; selected-checkpoint predictions, failure records, diagnostic panel dan resume state dapat divalidasi serta metrik dapat dihitung ulang.
- [ ] Analisis pascarun dan iterasi §12.3 memakai bukti train/validation, budget terbatas, parent/config diffs dan matched fairness; tidak menjamin kenaikan performa atau menuning ulang dari test.

Penutupan ilmiah mengikuti bukti: **apakah token akhir mempertahankan informasi terpilih, apakah metode membantu pada budget yang sama, apakah frozen LLM memanfaatkannya, serta bagaimana kualitas dan biaya berubah menurut budget dan pada subjek held-out**. Keempat pertanyaan dilaporkan terpisah agar hasil positif, parsial, negatif, dan inconclusive tidak saling tertukar.

## Pembaruan implementasi dan audit — 6 Oktober 2026

Contract suites untuk data, encoder, motion, kinematics, physical/language wiring, performance dan gate/artifact tersedia. Sembilan suite CPU/synthetic lulus pada 6 Oktober 2026, dengan [laporan dan source hashes tersimpan](../m4human_verification_20261006_hardening.json). Ringkasan perubahan ada pada [audit hardening](../m4human_hardening_audit_20261006.md). Pengukuran RTX3060/data aktual, tiny-overfit nyata, training ilmiah dan held-out results tetap pending.

Kandidat E128 mengganti baseline batch efektif E16 sehingga merupakan protokol baru dengan pilot/budget tercatat. Baseline M32, C/probe16 dan L16 dipertahankan dalam kandidat throughput. Record matched successful updates, skipped/retries, exposure, config/lineage, precision dan selection. Separate C clipping tetap wajib; loop C mengisi named norm compressor_fidelity/auxiliary before/after; maximum norm return helper tidak dibaca sebagai setiap norm grup. Native SDPA/explicit checkpoint pada L harus menjaga frozen eval dan input gradient.

Final E scientific eligibility berasal dari checkpoint mode/spec/decision dan readiness artifact hashes; smoke atau legacy tidak dapat dipromosikan dengan root threshold config. Threshold boleh ditentukan dari train/val sebelum selected run, tidak test. Artifact completeness terpisah dari hypothesis validity. Completion metrics/history memiliki LF-normalized UTF8 hash, legacy LF/CRLF equivalence hanya newline; exact-byte lineage tidak berubah. Packaging menolak run incomplete dan memvalidasi checksum ZIP ulang. Smoke historical bundle tetap incomplete jika checkpoint/predictions hilang, tanpa rekonstruksi bukti dari summary.

QA interval nominal mengidentifikasi query, label/evidence support menyebut actual valid frames. RPC XYZ/intensity saja untuk P0, RT optional ablation yang harus diaudit; tidak diasumsikan Doppler. [Runbook hardware/gates](../m4human_development_runbook.md#7-profiling-rtx-3060-12-gb-dan-gate-checkpoint) memuat PowerShell dan pre-test lock.

Optimasi RAM16GB/VRAM12GB menambah profiler M/C/probe/L, broad E batch sweep dan guard available RAM/CUDAheadroom dari persiapan hingga akhir run. QA/H/U metadata lazy dan index integer shared pada spawn; resource sum RSS bukan usage fisik. [Verifikasi terbaru dua belas suite](../m4human_verification_20261006_hardware.json) mencatat source/config hashes. Kandidat E128×1, M32×1, C/probe16×1, L4×4/generation8 tetap unmeasured; C/L dan probe harus memakai setting matched yang feasible pada semua kondisi. Readiness E terikat juga ke normalizer aktual. Benchmark RTX3060 dan scientific gates nyata masih harus dijalankan pada komputer dataset.
