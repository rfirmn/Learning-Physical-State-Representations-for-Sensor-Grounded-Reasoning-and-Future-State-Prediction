# 03 — Decoder Fisik: Supervisi Token Akhir dan Pembuktian Pelestarian Kinematik

Versi kontrak: `m4human_kinetok_v3`. Revisi: 4 Oktober 2026. Status: rencana penelitian/implementasi, bukan hasil training atau audit dataset. Acuan: [proposal](../proposal_riset_terbaru.md), [data dan encoder](01_encoder.md), [backbone gerak dan tokenizer](02_dynamic_model.md), [alignment bahasa](04_llm_layer.md), dan [evaluasi](05_evaluasi_testing.md). Semua nama model di sini adalah usulan implementasi lokal, bukan release opensource yang telah siap dijalankan.

## 1. Tujuan, hipotesis, dan batas kontribusi

Penelitian mengembangkan dan menguji pembentukan continuous motion tokens dari processed radar M4Human. Hipotesisnya: supervisi kinematik pada token akhir dapat mempertahankan informasi terpilih lebih baik daripada supervisi sebelum kompresi pada arsitektur/budget yang sama, dan informasi tersebut dapat dimanfaatkan frozen LLM. Kehilangan akibat kompresi belum dianggap terjadi; diagnostic sebelum/sesudah kompresi harus mengukurnya.

Decoder bukan keluaran bahasa, bukan world simulator, dan bukan forecasting model. Ia menyediakan target pelatihan dan alat pengujian kandungan representasi. Kinematika yang diuji ialah posisi joint/pelvis dan velocity turunan pada interval yang sudah diamati. Gaya, torsi, kontak, COM, niat, penyebab gerak, dan penguasaan hukum Newton berada di luar klaim.

Penanda: **F** fakta sumber primer; **T** identitas/derivasi; **R** desain beralasan; **H** default atau hipotesis belum tervalidasi. Semua kapasitas/bobot/threshold berikut R/H sampai audit dan pilot train/val menguncinya. Dokumen tidak menyatakan akses ke /dataset, training, runtime, atau hasil eksperimen telah terjadi.

### 1.1 Apa yang didukung literatur

- **F:** RadarLLM menunjukkan kelayakan trained radar-to-text dan memakai motion tokenization; itu bukan bukti frozen LLM memahami seluruh kinematika, maupun bukti bahwa kompresinya pasti kehilangan joint/velocity. [RadarLLM](https://arxiv.org/html/2504.09862v2).
- **F:** HuMoCon telah menggunakan velocity reconstruction untuk menjaga detail perubahan gerak. Jadi penambahan velocity loss saja bukan kebaruan kita. Temuannya tidak mengisolasi efek pengurangan token. [HuMoCon, CVPR 2025](https://arxiv.org/html/2505.20920v1).
- **F:** FiGMo melaporkan manfaat timestamp grounding; paper ini preprint Juni 2026, bukan bukti kausal kerusakan kompresi RadarLLM. [FiGMo](https://arxiv.org/html/2606.20888v1).
- **F:** SeMoCo, preprint, mendekode quantized tokens dengan joint-position reconstruction serta velocity/acceleration losses. Jadi supervisi fisik pada keluaran token melalui decoder sudah mempunyai pendahulu dekat; lokasi loss itu saja tidak disebut baru. [SeMoCo, §3.2](https://arxiv.org/html/2608.24334v2).
- **F:** MoTok, preprint, juga melatih motion-to-text captioner yang dikondisikan pada discrete motion tokens dari tokenizer frozen. Ia bukan hanya preseden generation tanpa jalur token-ke-bahasa. [MoTok, Appendix A.2/D.3](https://arxiv.org/html/2603.19227v1).
- **F/R:** Struktur joint–waktu MotionBERT adalah preseden bagi backbone lokal, bukan bobot siap transfer radar. [MotionBERT](https://arxiv.org/abs/2210.06551).

Kandidat kontribusi ialah kombinasi metode dan protokol pengujian pelestarian kinematik pada **U radar akhir di bawah budget token terbatas**, disertai pengujian pemanfaatannya oleh frozen LLM. Pembeda yang masih harus dibuktikan mencakup konteks sensor M4Human, continuous tokens, matched full-H fidelity dengan treatment lokasi supervisi, independent readability probes dan QA grounding. Tidak ada klaim pertama membuat tokenizer, kinematic loss, atau token-ke-bahasa. Kebaruan tetap harus ditinjau terhadap literatur; kemenangan baseline tidak otomatis memberi status metode baru, dan pergantian domain saja belum cukup untuk kebaruan metodologis kuat.

## 2. Representasi dan tahapan yang tidak boleh dicampur

Audit schema RPC, kalibrasi, pelvis/joint-map22, detik, gaps, split held-out subjects, dan sumber preprocessing dimiliki [01](01_encoder.md). Processed sekitar 50 GB bukan bukti kecukupan label atau fit resource. Inference memakai RPC XYZ/intensity hanya setelah channel terbukti; tidak mengasumsikan Doppler/SNR. GT/body params/annotation masks/action ID tidak menjadi input.

| Nama | Shape reference | Fungsi |
|---|---|---|
| P_enc / r_enc / f_enc | [B,32,22,3] / [B,32,3] / [B,32,256] | Prediksi encoder bersama; posisi meter |
| H | [B,32,23,128] | Rich hidden joint–waktu dari backbone gerak yang sama |
| M | [B,32,23,256], flattened [B,736,256] | concat(H[t,j],H[t,global]); rich precompression memory |
| Z | [B,32,256] | concat(masked mean H_joint128,H_global128); diagnostic sekunder |
| U | [B,K,256] | Token final satu per bin kronologis; K=16 primer, K=8/32 sekunder |
| P | [B,K,d_llm] | Projector(U); bukan joint position P_enc |

Ada 23 spatial tokens: 22 joint dan satu global token; jumlah ini bukan budget kompresi K. H/M mempertahankan sumbu serta provenance joint dan waktu, tetapi keberadaan sumbu tidak menjamin semua besaran fisik telah dipelajari. Z sudah joint-pooled, sehingga **tidak menjadi satu-satunya fidelity target**.

| Tahap | Diupdate | Frozen |
|---|---|---|
| E | Encoder dan position/root heads | Target recipe/split/normalizer |
| M | Common motion backbone + KinematicReadoutV2 | Encoder |
| C | Tokenlearner + full-H fidelity decoder + auxiliary readout, per kondisi | Encoder, motion/full-H physical head, H/M/Z reference |
| L | Projector independen per C_base/C_kin | Semua upstream/head dan Qwen2.5-1.5B-Instruct; tanpa LoRA |
| Probe | Probe diagnostik baru pada representasi frozen | Semua scientific checkpoints |

Tahap C menerima **cache H yang sama dan tetap** dari tahap M; M dibentuk deterministik dari H. Auxiliary heads adalah alat training, bukan bypass ke LLM. Jalur QA utama menerima U saja beserta masks/timestamps sensor yang identik. Gradient QA tidak memperbarui tokenizer pada protokol default.

## 3. Tiga decoder, tiga fungsi bukti

| Decoder | Input | Target | Batas bukti |
|---|---|---|---|
| Full-H physical readout | H + predicted-state residual pada tahap M | GT posisi dan filtered velocity | Menilai common motion estimator; tidak membuktikan informasi U |
| Common fidelity decoder | U + time/joint queries | Full frozen H, semua waktu/joint/global | Menjaga detail rich latent; H bukan trajectory GT |
| Auxiliary kinematic readout | M untuk C_base; U untuk C_kin | GT posisi dan velocity yang sama | Treatment lokasi supervisi; kualitas head training belum cukup untuk klaim token |

**R:** Full-H fidelity mengurangi risiko membuat baseline lemah karena hanya diminta merekonstruksi Z yang sudah menghilangkan sumbu joint. Namun fidelity H tidak menjamin sempurna mempertahankan semua semantik kinematik. Itulah alasan menambah physical supervision pada U dan memakai probes independen.

## 4. Full-H KinematicReadoutV2: pembelajaran gerak bersama

### 4.1 Struktur dan parameterisasi

```text
JointHead: H_joint128 -> LN128 -> Linear128->128 -> GELU -> Linear128->6
RootHead:  H_global128 -> LN128 -> Linear128->128 -> GELU -> Linear128->6
rows0:3 = normalized delta_position
rows3:6 = normalized velocity
```

JointHead digunakan bersama oleh 22 joint; RootHead memiliki parameter sendiri. Semua Linear memakai bias dan LN affine. Dua head memiliki 35.084 parameter menurut aritmetika rancangan (**T**), wajib dihitung ulang pada implementasi. Initializer hidden adalah Xavier-uniform/bias nol; LN weight satu/bias nol; final position rows nol dan velocity rows Normal(0,0,01)/bias nol (**H**). Posisi awal menyalin encoder; velocity menyediakan jalur gradien awal ke backbone yang harus diuji.

```text
p_H_relative = P_enc + s_delta_joint * delta_p_norm
r_H          = r_enc + s_delta_root  * delta_r_norm
v_H_relative = s_v_joint * v_joint_norm
v_H_root     = s_v_root  * v_root_norm
p_H_relative[...,pelvis,:] = v_H_relative[...,pelvis,:] = 0
J_global = p_relative + r
v_global = v_relative + v_root
```

Constraint pelvis melalui mask konstan differentiable, bukan mutasi in-place yang merusak autograd. Pelvis-zero dikeluarkan dari denominator relative error. Root=pelvis GT hasil joint map, bukan raw trans/COM. Keberhasilan posisi residual dapat berasal dari encoder; sebab itu full-H MPJPE rendah tidak dipakai sebagai bukti pelestarian U.

### 4.2 Scale dan loss tahap M

Train-only scales (**R/H**):
s_delta_joint/root = RMS komponen residual GT−predicted encoder, floor0.01m;
s_v_joint/root = RMS komponen filtered velocity, floor0.001m/s.
Tidak memakai mean output. Target kosong menghentikan estimasi scale; floor tidak menciptakan data.

```text
Lp_joint = masked_component_MSE((p_H_relative - p_GT_relative) / 1 metre)
Lp_root  = masked_component_MSE((r_H - r_GT) / 1 metre)
Lv_joint = masked_component_MSE((v_H_relative - v_GT_relative) / s_v_joint)
Lv_root  = masked_component_MSE((v_H_root - v_GT_root) / s_v_root)
L_M = Lp_joint + Lp_root + 0.5*(Lv_joint + Lv_root)/2
lambda_acceleration = lambda_bone = lambda_consistency = 0
```

Posisi memakai raw calibrated GT; velocity memakai recipe endpoint. Scale residual mengatur parameterisasi koreksi, bukan pengganti scale loss posisi1m. MSE komponen bukan MPJPE. Mask/count/reduction mengikuti bagian7. Bone length, v−D1(p), dan trapezoidal integration hanya diagnostics; reference GT sendiri dapat memiliki residual karena filtering/discretization.

Trunk/head dipilih dari validation posisi/root/velocity sesuai [02](02_dynamic_model.md), dibekukan bersama dan disimpan sebagai pasangan. Default bukan acceleration head; perluasan acceleration/sudut/orientasi membutuhkan label/reliability gate dan run terpisah.

## 5. Common full-H fidelity: baseline tidak boleh kehilangan target joint sejak awal

### 5.1 Query reconstruction decoder

**R/H:** kedua kondisi mempunyai instance decoder berarsitektur/initialization sama:

```text
query q[t,j] = PE256(relative_time[t]) + learned_joint_embedding[j]
j=0..22; t=0..31 -> [B,736,256]
input memory = exact U[B,K,256], token_mask, bin_time/provenance
PreLN query/memory -> CrossAttention(width 256,heads4) -> residual
LN -> FFN256->512->256/GELU -> residual -> final LN
Linear256->128 -> H_hat_normalized[B,32,23,128]
```

Satu cross-attention, tanpa self-attention antarqueries, dropout 0 reference. Linear memakai bias dan LN affine; initializer Linear/MHA Xavier-uniform dengan bias nol, LN weight satu/bias nol, dan learned joint embeddings Normal(mean=0, std=0,02). Initial state diklon antar kondisi. Query berisi waktu dan joint identity, **tidak** GT/predicted pose, target H atau label jawaban. Token/mask/time input sama dengan tensor yang tersedia untuk projector; tidak ada full-H attention memory tersembunyi.

PE memakai waktu sensor relatif dalam detik yang dinormalisasi nominal dt setelah audit; query joint-global memiliki index tersendiri. Recipes waktu/PE/hash dibagikan dengan02. Tidak memberi subject/action/annotation validity sebagai memory.

### 5.2 Target dan objective

Full H adalah keluaran backbone **detached/frozen**, bukan trajectory GT. Statistik channel train-only dari valid H semua waktu dan spatial tokens:

```text
mu_H[c] = mean_train_valid(H[...,c])
s_H[c]  = max(std_train_valid(H[...,c]), 1e-3)   # floor H
H_target_norm = (stop_gradient(H) - mu_H) / s_H
L_fidelity_H = mean_valid_t_j_c((H_hat_norm - H_target_norm)^2)
```

Normalizer sama bagi C_base/C_kin, dengan counts/hash; channel tanpa train support tidak diisi dari test. Gunakan population standard deviation (ddof=0), dihitung streaming fp64 pada train dengan mask sumber tahap C yang sama. Validity berasal sensor/time, mencakup global token dan seluruh joint yang sah pada common frame support di bagian 7. Full-H reconstruction tidak otomatis membuktikan joint identity semantik tersimpan; periksa rekonstruksi per joint/waktu. Untuk metrik raw latent, inverse-transform H_hat = mu_H + s_H * H_hat_norm.

Z-reconstruction boleh menjadi secondary diagnostic/ablation bila dibenarkan, tetapi **bukan pengganti primary H fidelity**. M dibentuk dari H; tidak wajib merekonstruksi kedua bagian concatenation M yang redundan.

## 6. Auxiliary kinematic head: treatment M versus U

### 6.1 Satu arsitektur, lokasi input berbeda

`TokenKinematicReadoutV1` (**R/H**, nama lokal) memakai time+joint queries 32×23, width 256, 4 heads, satu cross-attention, FFN256→512→256, final LN dan Linear256→6. Struktur query/normalization sama dengan decoder fidelity, tetapi parameter independen dan output 6/query. M pada cache ialah M_raw=concat(H_joint,H_global). Source attention baseline/probe ialah M_attn=M_raw+PE_waktu dari helper tunggal dokumen 02, dengan mask sumber yang sama dengan compressor. PE waktu ditambahkan tepat sekali; tidak ada tambahan joint embedding pada memory karena H telah memuat identitas joint. Learned joint embedding hanya pada query decoder. Pada source U, memory tetap tensor U akhir yang telah memiliki encoding bin, tanpa PE ulang; tidak mengakses H/M atau states untuk melengkapinya. Head membaca seluruh memory sumber yang diizinkan:

| Kondisi | Memory auxiliary | Gradien auxiliary |
|---|---|---|
| C_base | M_attn flattened[B,736,256], view sumber compressor yang sama | Head saja; H/M fixed sehingga tidak mencapai compressor |
| C_kin | U[B,K,256], bin times dan token_mask yang sama dengan projector | Head **dan** compressor |
| Kedua kondisi | Common fidelity memory U, target full H | Fidelity decoder dan compressor |

Kapasitas parameter auxiliary head sama; input length berbeda, sehingga FLOPs/activation tidak diklaim sama. Kedua head berinitialization sama dan mendapat raw label/target/scales identik. Compressor C_base tetap belajar dari full-H fidelity. **Ini bukan baseline tanpa physical training:** common upstream sudah physically supervised, full-H fidelity memuat representasinya, dan auxiliary baseline mendapat label yang sama.

Auxiliary output adalah absolute physical reconstruction, bukan residual bypass:

```text
p_relative_hat = 1 metre * output_joint[...,0:3]
r_hat          = 1 metre * output_global[...,0:3]
v_relative_hat = s_v_joint * output_joint[...,3:6]
v_root_hat     = s_v_root  * output_global[...,3:6]
pelvis relative p/v = 0
L_aux = Lp_joint + Lp_root + 0.5*(Lv_joint + Lv_root)/2
L_C = L_fidelity_H + lambda_aux * L_aux      # lambda_aux=1.0 H
```

Lp memakai scale1m dan Lv scale train s_v dari bagian4, target recipe/masks yang sama. Output physical head dihitung fp32; final Linear memakai Xavier-uniform/bias nol seperti bagian 5, bukan posisi residual zero-init. Tidak ada P_enc/r_enc/features/GT skip ke head U. Label anatomy/time query boleh; nilai posisi/velocity target tidak boleh.

Walau rumus total loss sama, `grad_C(L_aux)=0` pada C_base dan umumnya nonzero pada C_kin. Log gradien tiap term di smoke check. Jangan menyatakan kedua kondisi memperoleh gradient signal identik, ataupun bahwa perbandingan mengisolasi pure architecture: ia menguji **penempatan physical supervision sebelum versus setelah kompresi**.

### 6.2 Rekonstruksi retrospektif, bukan forecasting atau online per-frame

Fidelity dan auxiliary query pada waktu t dapat membaca seluruh U dari 32-frame observed window. Mereka merekonstruksi semua waktu teramati, termasuk waktu awal. Ini retrospective window reconstruction, bukan streaming causal output pada tiap t dan bukan future prediction. Sensor encoder/backbone tetap causal; tidak ada input sensor di luar observed support bersama. Prefix causality test berlaku untuk H, bukan decoder U yang memang mempunyai akses seluruh window.

### 6.3 Budget dan selection tahap C

Default reference belum dijalankan (**H**): K=16 primer; K=8/K=32 sekunder. Width256, arsitektur compressor/head, fixed upstream checkpoints, initialization, optimizer step cap, sample order, target recipe dan split sama pada paired conditions. Setiap budget mempunyai paired run dari initial state yang sama; tidak fine-tune satu budget dari pemenang budget lain tanpa protokol terpisah.

Default tahap C mengikuti dokumen 02: AdamW learning rate 3e-4, betas 0.9/0.999, weight decay 0.01 untuk matrix weights saja, microbatch 2 dan accumulation 8 (effective batch 16), serta cap 30 epoch. Mulai microbatch 1 untuk profiling bila diperlukan; ubah effective batch kedua kondisi secara sepadan sebelum lock. Semua angka adalah H, bukan jaminan muat. Warm-up 5% successful optimizer steps, cosine schedule, gradient clipping 1, FP16 AMP+GradScaler bagi neural attention; target/loss fp32 dan label offline fp64. Final paired runs memakai successful-update cap yang sama; early stopping sepihak tidak menggantikan exposure matched.

Gradient clipping tahap C mengikuti dokumen 02: norm 1 untuk grup compressor+fidelity decoder dan norm 1 terpisah untuk auxiliary head, bukan satu norm gabungan. Fixture memastikan gradien auxiliary baseline tidak memengaruhi update compressor melalui faktor clipping bersama. Initializer q0 dan joint query serta grouping/checksum menjadi bagian config snapshot.

Selection rule sama untuk kedua kondisi: minimal validation normalized full-H fidelity, tie-break epoch lebih awal; physical validation tetap dilaporkan tetapi tidak dijadikan keuntungan selection khusus C_kin. Alternatif selection yang sah harus pre-registered pada train/val bersama, bukan dipilih dari QA test. Batas efek praktis dan tolerance mengacu pada dokumen 05; tidak ada target kenaikan akurasi yang dianggap sudah tercapai.

Auxiliary memory M yang panjang dapat meningkatkan biaya attention: ukur microbatch kecil; query chunking memadai karena tidak ada query self-attention. Chunking harus lolos equivalent forward/backward dan loss-denominator tests, tidak mengubah objective. Jika resource membatasi studi, kunci subset contiguous/budget feasibility bersama sebelum test. Jika hanya K16 berjalan, laporkan single-budget; tidak mengklaim trade-off multi-budget.

## 7. Validity, losses, dan numerical boundaries

Sensor validity dan annotation validity berbeda; mask target **tidak** menjadi input ke encoder/backbone/compressor/head/LLM.

| Mask | Semantik |
|---|---|
| sensor_position_valid_joint/root | RPC/context/time yang sah; joint reference broadcast frame, bukan confidence anatomis |
| annotation_position_valid_joint/root | Target posisi sah; relatif memerlukan pelvis GT |
| predicted_derivative_valid_joint/root | Tujuh source positions sensor-valid dan waktu valid; endpoint≥6 |
| target_derivative_valid_joint/root | Tujuh GT positions valid dan waktu valid; endpoint≥6 |
| loss_position_valid | Intersection sensor dan annotation untuk kelompok yang bersangkutan |
| loss_velocity_valid | Intersection predicted dan target derivative support |
| latent_fidelity_valid | Sensor/time support H, per joint/global; tidak memakai GT eligibility |
| global p/v valid | Intersection relatif dan root field yang digunakan |

Mask sampling/packing sama bagi semua conditions. Kegagalan numerik model tidak boleh mengecilkan cohort atau token mask agar metric membaik. Non-finite prediction pada sensor-valid support = failure; invalid padded entries di-zero sebelum arithmetic.

**Kesetaraan sumber tahap C/probes:** common_frame_valid = feature_valid AND root_sensor_valid AND all(joint_sensor_valid) AND context_time_valid. Memori M yang benar-benar diizinkan bagi compressor, auxiliary C_base, dan probe precompression memakai latent_valid AND common_frame_valid; jangan memberi probe M partial frames yang sengaja tidak diterima compressor. Fidelity target dan statistik mu_H/s_H memakai intersection yang sama. Query decoder tetap berjumlah 32×23, tetapi target yang tidak eligible tidak dihitung sebagai rekonstruksi yang salah atau pengamatan nol.

Untuk physical auxiliary/probes tahap C, posisi pada endpoint e memerlukan common_frame_valid[e] dan annotation position validity; velocity memerlukan seluruh tujuh common frames pada e−6..e serta target derivative validity. Common support yang sama dipakai untuk semua source M/U dan kedua kondisi. Full-H physical readout tahap M boleh memakai per-joint support-nya sendiri: itu estimand upstream terpisah, bukan alasan memberi precompression probe informasi tambahan. QA tetap mengikuti cohort eligibility bersama yang dikunci dan melaporkan kegagalan sensor, tidak mengecilkan panel berdasarkan hasil model.

Masked component MSE memilih valid XYZ terlebih dahulu lalu membagi jumlah komponen actual tiap kelompok. NaN×0 bukan masking. Pelvis relatif tidak masuk denominator; root/global memiliki term sendiri. Empty group memberi finite graph-connected zero dan count0 tanpa label nol palsu; faktor velocity/2 tidak diam-diam berubah karena satu kelompok kosong. Batch tanpa semua target valid tidak melakukan optimizer/scheduler step. Untuk fidelity, valid H sendiri wajib tersedia agar compressor dapat belajar; coverage physical targets tetap dilaporkan.

Accumulation memakai ukuran grup aktual, weighting/count policy terpin, unscale sebelum clip, scheduler hanya maju pada successful update; catat overflow/skipped updates. Resume memulihkan optimizer/scaler/scheduler/RNG/sampler dan hash sama. BF16 hanya alternatif setelah smoke device/software; precision aktual tercatat. Gradient language melalui frozen LLM tetap perlu autograd ke projector, bukan no_grad menyeluruh. [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

## 8. Reference derivative dan pembanding matematis

### 8.1 Recipe causal shared

`kinematic_recipe_v3` menunjuk derivative subrecipe `aligned12hz_endpoint_sg_v2`; operator tidak berubah hanya karena kontrak kompresi direvisi. Raw calibrated position tetap immutable. Velocity reference adalah estimasi filtered, bukan pengukuran Doppler langsung.

```text
untuk endpoint e>=6:
tau_i = time_s[i] - time_s[e], i=e-6..e
A_i = [1,tau_i,tau_i^2]
beta = least_squares(A,x)
D1(x)_e = beta_1
D2(x)_e = 2*beta_2            # optional, bukan target P0
```

**T/F:** pada grid uniform12Hz, gunakan chronological endpoint coeffs `savgol_coeffs(7,2,deriv=1,delta=1/12,pos=6,use="dot")`. Tidak centered filter, membalik coefficients atau membagi dt dua kali. [SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html). Pure-PyTorch dapat memakai least squares; SciPy reference bukan dependency native wajib.

Actual irregular timestamps memakai fit waktu nyata fp64 dengan rank/conditioning audit. Gap/reset/duplicate time memutus support. Tanpa timestamp, nominal frame-ID/12 hanya setelah audit; tidak membuat measured timestamps palsu. First6 setiap L=32 window derivative-invalid walau cache recording punya history. Target/QA tidak memakai GT sebelum window; encoder context tiga frame sebelum window dicatat sebagai shared sensor support.

### 8.2 Comparator yang tidak menjadi ablation kompresi

| ID | Sumber | Tujuan |
|---|---|---|
| D_encoder_sg | Raw P_enc/r_enc + D1 dari predicted history | Mathematical/direct-rule reference |
| D_learned_kinematics | Full-H p/r/v tahap M | Diagnosis kualitas common motion backbone |
| D_refined_sg | Posisi full-H decoder + D1 | Diagnosis refinement versus learned velocity |

Recipe/time/support/masks/target panel sama. AI juga menerima f_enc sehingga perbandingan ini bukan isolasi kapasitas/information access. Ia tidak menggantikan paired C_base/C_kin sebagai pembuktian tokenizer. Jangan memakai derivative posisi yang sudah difilter dua kali; filtered position D0 boleh dilaporkan terpisah. Jika matematika lebih baik, laporkan tanpa memaksakan kemenangan AI; hypothesis pelestarian token tetap diuji pada upstream yang sama.

## 9. Target tugas: sederhana, terukur, dan reliable

Satu registry/recipe untuk generator QA, probes, direct-rule, parser dan evaluator; jangan menyalin implementasi formula. Query bahasa menyebut interval/body_part, bukan action ID. GT membuat target; sensor-derived prediction memakai recipe sama untuk evidence. Evidence bukan label GT yang diberikan kepada LLM.

Default (**H**): observation interval t0..t31; derivative task interval t6..t31. Pada12Hz, observed endpoint span31/12≈2.58s; first valid derivative di0.5s. Maksimal26 eligible endpoint. Coverage0.75 berarti minimal20/26 setelah guards, bukan75% dari jumlah yang kebetulan tersisa.

| Task | Enum | Scope |
|---|---|---|
| root_speed_trend | speeding_up / slowing_down / no_overall_trend / unknown | P0: tren keseluruhan speed pelvis global |
| relative_limb_motion | faster_left / faster_right / tie / unknown | P0: arms atau legs relatif pelvis |
| limb_onset_order | left_before_right / right_before_left / approximately_simultaneous / unknown | Opsional; diperlukan bila mengklaim urutan peristiwa |
| root_radial_direction | approaching / receding / stationary / unknown | Opsional, setelah origin/range audit |

### 9.1 Root speed trend

s[t]=norm(v_root[t]). Fit slope dengan median pasangan (s[j]−s[i])/(time[j]−time[i]), i<j, kemudian intercept median(s−slope*time). Default H deadband slope0.05m/s²; residual MAD maksimal0.10m/s. Coverage gagal atau fit tidak stabil →unknown; pada support/fit sah, slope>deadband speeding_up, slope<−deadband slowing_down, lainnya no_overall_trend. Slope speed bukan norm vector acceleration; gerak melingkar uniform dapat berubah arah tanpa speed meningkat.

Task ini mengukur **tren keseluruhan**. `speeding_up`/`slowing_down` tidak mensyaratkan perubahan monoton setiap frame. `no_overall_trend` berarti slope berada dalam deadband pada support/fit sah, bukan speed selalu konstan; perubahan naik lalu turun dapat memperoleh label ini. `unknown` tetap menandai support atau kualitas fit yang tidak memadai, bukan tren kecil yang sah. Pertanyaan kanonis: “Selama interval ini, apakah speed pelvis secara keseluruhan cenderung meningkat, menurun, atau tidak menunjukkan tren naik/turun yang cukup kuat?” Registry, prompt dan parser menggunakan empat enum pada tabel tanpa mengubah rumus, deadband atau menambah kelas.

Fixture task sintetis untuk implementasi memakai 26 endpoint valid pada t=(6..31)/12 detik, dengan speed dalam m/s: sepuluh nilai 0.2, lalu [0.3,0.5,0.8,0.8,0.5,0.3], lalu sepuluh nilai 0.2. Pada input speed hasil derivative ini, median slope=0 dan residual MAD=0, sehingga label yang diharapkan `no_overall_trend` walau speed mencapai 0.8 m/s. Fixture ini memeriksa makna agregasi task, bukan kualitas derivative atau hasil M4Human. Fixture speed konstan juga menghasilkan `no_overall_trend`; coverage tidak memadai menghasilkan `unknown`.

### 9.2 Relative limb motion

Audited arms=elbow+wrist tiap sisi, legs=knee+ankle. Gunakan intersection endpoint semua joint kiri/kanan sah, bukan cohort per sisi berbeda:

```text
speed_left  = mean_joint_left(median_valid_time(norm(v_relative[t,j])))
speed_right = mean_joint_right(median_valid_time(norm(v_relative[t,j])))
difference = speed_left - speed_right
```

Default H tie margin0.05m/s. Difference>margin faster_left, <−margin faster_right, selainnya tie hanya bila support reliable; support kurang →unknown. Tie bukan abstention. Translasi pelvis sengaja tidak dihitung sebagai artikulasi limb. Keunggulan task ini tidak membuktikan semua bentuk koordinasi telah dipahami.

### 9.3 Limb onset order: gate terpisah, bukan klaim otomatis

Reference H membandingkan group-speed kiri/kanan pada body_part arms/legs, `g_side[t]=mean_j(norm(v_relative[t,j]))`, memakai groups yang sama dengan9.2. Ini urutan onset gerak relatif pelvis, bukan hand–torso event dari kontrak lama; labels lama tidak direlabel tanpa regenerasi.

Syarat label: seluruh 26 endpoint kedua kelompok valid dan contiguous, semua derivative supports sah. Default H on0.15m/s, off0.10m/s, quiet3endpoint, confirmation3endpoint, simultaneous margin2/12s. Threshold bukan nilai resmi M4Human.

Finite-state recipe:
1. Tiga endpoint pertama interval harus quiet≤off pada kedua sisi. Active/intermediate/missing awal →unknown/left_censored; tidak menunggu quiet lalu memilih onset episode berikutnya.
2. Setelah armed, crossing≥on menjadi candidate; tiga endpoint on berurutan mengonfirmasi. `onset_time_s` ialah crossing, `confirmed_at_s` waktu konfirmasi. Nilai intermediate sebelum candidate tidak memulai event.
3. Spike gagal dicatat; re-arm membutuhkan3quiet. Yang dicari first confirmed onset; gap tidak disembuhkan dengan episode berikutnya.
4. Candidate yang belum terkonfirmasi saat interval berakhir →unknown/right_censored; tidak membaca future frame. Tanpa dua onset reliable →unknown, bukan mengarang order.
5. delta=onset_left−onset_right: <−margin left_before_right; >margin right_before_left; selainnya approximately_simultaneous jika kedua onset sah.

Filter/confirmation mempunyai lag dan batas resolusi. Audit label independen pada train/val memeriksa synthetic known-order, label stability windows5/7/9, confirmation2/3/4, sampled trajectory inspection, coverage/censoring, dan noise/alignment. Jika gate gagal, jangan masukkan task ke klaim event-order. Root_speed_trend+relative_limb_motion tetap menjadi scope PoC, tetapi tidak disebut bukti lengkap urutan peristiwa.

### 9.4 Root radial direction opsional

Origin radar o terkalibrasi; range rho=norm(r−o); radial velocity=dot(v_root,r−o)/rho untuk rho>guard. Default H range guard0.05m, deadband0.05m/s, fraction minimum0.80. Proporsi mendekat/menjauh/deadband dihitung pada final guarded support≥20/26; mixed direction →unknown. Range singular →null evidence, bukan v0 palsu. Stationary di sini berarti radial kecil, bukan seluruh tubuh diam atau tidak ada translasi tangensial.

### 9.5 Status target, evidence, dan parsing

| Keadaan | Representation/penilaian |
|---|---|
| Target answerable | target_status=defined, validity=valid, answer enum nonunknown |
| Unknown yang dapat diaudit | target_status=unknown, validity=unknown, answer=unknown; policy panel dipatok train/val |
| Annotation tidak cukup/audit gagal | target_status=undefined, validity=excluded, answer=null, reason; coverage dilaporkan |
| Evidence numerik undefined | evidence_status=undefined, numeric=null; tidak otomatis membuat seluruh target excluded |
| Model abstention | Jawaban unknown pada target answerable tetap dinilai terhadap target asli |
| Parse failure | parse_valid=false dan dihitung salah; tidak diubah menjadi unknown sah |

Alias target_status/validity one-to-one, evaluator menolak konflik. Missing GT bukan stationary dan bukan otomatis unknown reliable. Annotation eligibility tidak dikirim sebagai feature/prompt. Simpan count sebelum/sesudah filter per task/subject/recording, unknown/parse/excluded rates.

```text
QARecord:
  qa_id, window_id, recording_id, source_keys, frame_ids, time_s
  observation_interval_s, task_interval_s, task, body_part?, question_id, question
  answer, target_status, validity, evidence_status, reason
  label_support, evidence_support, label_recipe_version, derivative_recipe_id
  recipe_hash, joint_map_hash, split_hash, gt_source_hash

EvidenceRecord:
  window_id, source=gt_reference|sensor_prediction
  coordinate_frame, units, task_interval_s, support_frame_ids
  numeric_aggregates, diagnostic_answer, evidence_status, reason
  valid_count, eligible_count, onset_time_s?, confirmed_at_s?, censor_reason?
  encoder_hash, motion_hash, decoder_or_probe_hash, recipe_hash
  uncertainty=null  # sampai benar-benar dikalibrasi
```

Evidence_support menunjuk label_support yang sama. Valid fraction bukan posterior. Attach predicted evidence tidak menimpa raw LLM answer; rule correction/oracle adalah kondisi terpisah. Angka yang diucapkan LLM bukan otomatis ukuran fisik sensor.

## 10. Token U, probes independen, dan kontrol

### 10.1 Interface kompresi yang wajib sama

`KinematicTokenLearnerV1` diatur pada dokumen 02: learned bin-restricted query cross-attention atas M; satu token 256/bin, query tidak tergantung pertanyaan. Bin K8/16/32 kronologis pada L=32; K=16 primer. Semua conditions memakai bin boundaries, timestamp encoding, support IDs dan masks sensor yang sama. Tidak ada duplikat S/S, slot f_enc tambahan, GT/action metadata, atau raw predicted-state bypass.

Bin kosong mengeluarkan content/token nol dengan token_mask false, termasuk setelah PE; attention all-invalid ditangani finite tanpa softmax atas semua−inf. Bin bounds/time berasal seluruh source timestamps bukan subset model yang berhasil. Coverage sensor policy dikunci bersama sebelum test; numerical failure tidak dikeluarkan condition-specifically.

U yang diuji adalah **tensor persis** sesudah final token/time encoding dan sebelum projector LN; log U hash/recipe. Projector diatur pada dokumen 04: LN256→Linear512→GELU→Linear d_llm, d_llm dibaca config Qwen; reference 1536. Semua upstream/head/LLM frozen saat projector training, tanpa LoRA.

### 10.2 Pengukuran kandungan, bukan hanya kualitas training head

Setelah checkpoints frozen, latih ulang physical probes scratch pada M dan masing-masing U; Z sekunder. Pakai arsitektur temporal-joint-query `TokenKinematicReadoutV1` yang sama, initialization, targets, masks, scales, updates dan validation rule. Tidak reuse training auxiliary head atau memberi shortcut state. Input width sama 256; panjang sekuens 736/32/K berbeda dan cost dicatat. Full-H direct-head diagnostic terpisah karena width/topology/bypass berbeda.

Laporkan:
- MPJPE relative/global/root trajectory error dalam m/mm, tanpa root/Procrustes alignment pada global.
- Mean velocity vector error dan speed error dalam m/s, per kelompok/root dan gerak cepat strata.
- Task-probe accuracy/macro-F1 dari prediksi posisi/velocity probe fisik independen melalui recipe deterministik yang sama, tanpa LLM.
- Per-joint/time normalized H reconstruction, selected-task coverage, unknown dan sensor failures.
- LLM structured-answer macro-F1/accuracy, parse-invalid sebagai salah, serta paired differences sesuai05.

Alur task-probe utama adalah `M/U → probe fisik independen → prediksi posisi/velocity → recipe deterministik → jawaban task`. Task/body_part menentukan agregasi recipe, bukan masukan tambahan ke probe fisik yang tetap memakai query waktu/joint. Classifier task langsung bukan bagian P0; jika kelak diperlukan sebagai diagnosis underfit, kapasitas/query/task input harus matched dan hasilnya dilaporkan terpisah, tanpa mengganti primary setelah melihat test.

**Batas inferensi:** probe readability bergantung kapasitas/optimasi; probe gagal tidak membuktikan information-theoretic absence. M→U gap harus diperiksa terhadap noise label, upstream quality, timestamp/mask alignment dan convergence. Menurunnya QA saja tidak membuktikan kehilangan akibat kompresi. Menguatnya physical probe tanpa QA gain berarti manfaat bahasa belum terbukti.

### 10.3 Kontrol yang tetap dalam scope

Kontrol murah: train majority, text-only LLM, direct-rule dari predicted states/full-H output, dan probes tanpa LLM. Grounding P0 menggunakan pasangan source A dan donor B yang berbeda recording tetapi beraktivitas sama, dengan reference answers answerable yang **berbeda** dan terverifikasi. Task/body_part/pertanyaan identik, coordinate/joint recipe sama, interval dan dukungan waktu kompatibel. Jika interval numerik dalam pertanyaan tidak berlaku bagi kedua source, pasangan tidak eligible; jangan mengubah satu pertanyaan atau memaksakan label asli yang support-nya berbeda. Action ID hanya untuk stratifikasi donor, bukan input.

Untuk tiap pasangan, jalankan U_A dan U_B dengan pertanyaan yang sama, simpan kedua reference answers, raw predictions serta provenance. Definisikan `c_A`/`c_B` sebagai prediksi parsed-valid yang tepat terhadap reference masing-masing:

- **PairBothCorrect** = proporsi seluruh eligible pairs dengan c_A AND c_B; parse failure dihitung salah.
- **CorrectChange** = proporsi seluruh eligible pairs dengan kedua jawaban benar dan jawaban A berbeda dari B. Karena reference answers sudah berbeda, metrik ini setara dengan PairBothCorrect, bukan bukti independen kedua.
- **ConditionalCorrectChange** = diagnostic opsional: jumlah pasangan dengan kedua jawaban benar dibagi jumlah pasangan dengan c_A. Denominator awal-benar dan coverage-nya dilaporkan; bila nol, hasil undefined. Nama ini tidak boleh dipakai sebagai alias CorrectChange primer.
- **coverage** = paired windows/eligible query windows, count pairs per task/subject/recording dan jumlah awal-benar; simpan reasons pasangan tidak tersedia/ditolak. Jangan menyimpulkan grounding luas dari coverage kecil.

PairBothCorrect/CorrectChange memakai denominator pasangan eligible yang tetap. ConditionalCorrectChange menunjukkan respons pada kasus awal telah benar tetapi tidak menggantikan metrik primer; RawChangeRate hanya diagnostic perubahan tanpa jaminan benar. Donor same-label hanya **invariance control**, bukan bukti perubahan jawaban mengikuti gerak. Pemilihan donor berdasar reference/support yang dikunci sebelum test inference, bukan berdasar apakah model benar. Nama dan formula mengikuti evaluator dokumen 05.

Shuffled/dropped U adalah representation perturbation yang dapat OOD; penurunan skor tidak sendirian membuktikan physical reasoning. Arbitrary latent intervention bukan physical counterfactual. Oracle GT/text facts terpisah jelas, bukan kondisi deployment. C_pool/C_state hanya optional mechanistic controls dengan confounds dinyatakan; tidak otomatis menambah scope P0.

## 11. API, lineage, resource, dan acceptance checks

### 11.1 API rencana, bukan script yang sudah tersedia

```text
KinematicReadoutV2.forward(H, P_enc, r_enc, sensor_masks, scales) -> PhysicalPrediction

FullHFidelityDecoderV1.forward(U, token_mask, bin_time_s,
                              query_time_s, joint_map) -> H_hat_norm

TokenKinematicReadoutV1.forward(memory256, memory_mask, memory_provenance,
                               query_time_s, joint_map, scales) -> PhysicalPrediction
# memory256 = M_attn pada C_base atau exact U pada C_kin; tanpa state/GT arguments

build_reference_derivatives(position, time_s, source_validity, recipe)
  -> velocity, derivative_valid, support_metadata

build_evidence(physical_values, time_s, source_validity, recipe, task_definition)
  -> EvidenceRecord
```

Implementasi Pure-PyTorch dan semua Python via .venv/uv; reuse helper/report/checkpoint jika semantics sesuai. Read-only LMDB; label/normalizer/cache/checkpoint baru hanya di derived root, bukan source. Tidak perlu decoder framework, database baru, compiled point ops, atau inference SMPL-X. Sanity sebelum training panjang sesuai SOP.

Config authority minimal:

```yaml
contract_version: m4human_kinetok_v3
upstream: {encoder: M4HumanSetEncoderV2, motion: CausalDSTformerLiteV1, frozen: true}
memory: {length: 32, spatial_tokens: 23, h_width: 128, m_width: 256}
compression:
  model: KinematicTokenLearnerV1
  primary_budget: 16
  secondary_budgets: [8, 32]
  token_width: 256
  per_bin_tokens: 1
fidelity:
  model: FullHFidelityDecoderV1
  target: full_frozen_H
  queries: time_and_joint_all_observed
  query_width: 256
  heads: 4
  output_width: 128
  normalization: train_channel_mean_std
  z_fidelity: secondary_only
auxiliary:
  model: TokenKinematicReadoutV1
  baseline_source: fixed_M_attn
  proposed_source: U
  query_width: 256
  heads: 4
  output_width: 6
  state_bypass: false
  reconstruction: retrospective_observed_window
  lambda_aux: 1.0
physical_loss: {position_scale_m: 1.0, velocity_scale: shared_train_rms, velocity_weight: 0.5}
recipe: {label: kinematic_recipe_v3, derivative: aligned12hz_endpoint_sg_v2}
precision: {neural_reference: fp16_amp_grad_scaler, physical_loss: fp32, resolved: null}
assets: {schema_hash: null, coordinate_hash: null, joint_map_hash: null,
         encoder_hash: null, motion_hash: null, h_cache_hash: null,
         split_hash: null, normalizer_hash: null, recipe_hash: null}
```

Null audit fields menutup scientific run sampai diisi. Resume/checkpoint loading strict dan trusted tensor policy; jangan unrestricted pickle fallback. Full-H backbone+physical-head disimpan bersama; tahap C bundle berisi compressor/fidelity/aux heads plus fixed-upstream hashes. Projector menunjuk tokenizer version tepat, tidak latent checkpoint asing.

### 11.2 Cache dan aritmetika

Cache H harus berasal dari window L=32 dengan reset, context, dan positional policy yang sama dengan training; satu forward seluruh recording tidak otomatis ekuivalen. Cache H/Z dipakai bersama lintas kondisi/budget; M dibentuk saat dibaca. Gunakan streaming shards/memmap, tanpa preload seluruh data ke RAM 16 GB atau duplikasi per paraphrase QA. Namespace target GT/annotation terpisah. Tensor detached biasa dibaca dari cache; inference_mode tensors tidak langsung dipakai untuk operasi yang perlu menyimpan tensor bagi autograd.

| Payload satu window | Ukuran, di luar metadata |
|---|---|
| H fp16 / fp32 | 184 / 368 KiB |
| M fp16 / fp32, on-the-fly | 368 / 736 KiB |
| Z fp16 | 16 KiB |
| U8 / U16 / U32 fp16 | 4 / 8 / 16 KiB |
| Physical p/r/v fp32 | 17.25 KiB, tanpa duplicate global joints |

Contoh 30.000 windows: H+Z fp16 = 6.144.000.000 byte, sekitar 6,14 GB desimal, belum masks/GT/container; jumlah window aktual belum diaudit. Memori training juga meliputi attention/query activations, gradients, Adam states, scaler, allocator, dan I/O. Frozen LLM tetap memerlukan activations untuk gradient projector. Payload bukan peak VRAM, dan runtime tidak dapat diperkirakan hanya dari ukuran dataset 50 GB. Profil CPU/RAM/disk/GPU dilakukan pada RTX 3060 dengan microbatch kecil dahulu.

Lineage minimal: source/window/frame/time/context IDs, recording/split/masks, schema/coordinate/joint-map, encoder/motion/full-H head, H cache, compressor/fidelity/aux/readout/probe hashes, normalizer/recipe, K/bins/encoding, seed/precision/software. Artefak incomplete tidak dianggap selesai; config/recipe berubah berarti run/cache baru. Jangan menghapus laporan MM-Fi historis atau docs/report_training.

Readout/probe reporting mengikuti [dokumen 05 §12.2–12.3](05_evaluasi_testing.md#122-kontrak-log-minimum-untuk-analisis-pascarun): per-term p/v/fidelity losses, units/error sums/counts per-window/joint, task answers dari recipe dan fixed train/val diagnostic trajectories. Fresh probe convergence dan residual bypass diperiksa sebelum atribusi compression loss. Semua iterasi menggunakan train/validation serta provenance baru; kualitas log mendukung diagnosis, bukan jaminan peningkatan.

### 11.3 Satu runnable self-check yang harus ada saat implementasi

| Check | Bukti wajib |
|---|---|
| Input–target isolation | GT/body/action/annotation-mask perubahan tidak mengubah inference/U |
| Units/pelvis | Global=relative+root, pelvis relative p/v0, m↔mm dan m/s benar |
| Derivative | Constant/linear/quadratic pada uniform/irregular time; first6 invalid; no future/gap support |
| Full-H readout | Residual init menyalin encoder, velocity memberi gradien ke backbone |
| Fidelity target | H fixed dan lengkap32×23; semua channel/joint/global masked benar; Z bukan target tunggal |
| Auxiliary placement | Same params/labels; C_base aux gradients hanya head; C_kin aux gradient mencapai compressor |
| Retrospective access | Query early-time boleh membaca seluruh U; tidak mengklaim causality head |
| Masks/numerics | Empty bin/all-invalid finite, NaN valid gagal, padding tidak menjadi stationary |
| Shared source | Same H/cache/window/support/initialization/budget pada kondisi berpasangan |
| Probes | Fresh independent matched probes M/U; no GT/P_enc/fullH bypass ke U |
| Task reliability | Trend/limb fixtures, termasuk naik–turun dengan no_overall_trend dan support kurang dengan unknown sesuai §9.1; optional onset censor/quiet/spike/confirmation/simultaneous |
| Freeze integration | Tahap L optimizer hanya projector; upstream/LLM hashes tetap, projector grad nonzero |
| Resource | Actual peak RAM/VRAM, throughput, resolved precision; belum feasibility claim |

## 12. Keputusan ilmiah dan handoff

Kunci tasks/thresholds/budgets/splits/selection/parser/toleransi efek praktis sebelum test. P0 satu seed adalah exploratory; interval ketidakpastian paired per subject/recording cluster, bukan independent-frame CI. Banyak seeds diperlukan bila ingin klaim robust terhadap initialization, bukan requirement otomatis PoC.

Hasil dibaca berlapis: apakah informasi tersedia pada M, apakah kompresi/proposed treatment mengubah readability U, lalu apakah frozen LLM memanfaatkannya. Physical preservation tanpa QA gain adalah hasil terbatas; baseline tanpa gap berarti dugaan bottleneck tidak didukung pada kondisi itu. Negative result sah hanya bila target reliable, training memadai, comparator adil dan uncertainty cukup informatif. Bug/optimasi gagal/schema salah →inconclusive, bukan penolakan hipotesis.

Handoff 03: audited immutable positions + derivative recipe, shared scales/masks/task registry, full-H physical checkpoint, fixed H cache, paired trained tokenizers/fidelity/aux heads, independent U probes, provenance dan actual profile. Handoff 04 hanya U+sensor masks/time/provenance serta frozen hashes; bukan GT/evidence jawaban. Klaim dibatasi pada kinematika/tasks yang teruji, bukan seluruh fisika manusia.
