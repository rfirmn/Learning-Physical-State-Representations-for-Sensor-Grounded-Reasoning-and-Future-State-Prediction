# 02 — Representasi Gerak dan Tokenizer dengan Pelestarian Kinematik

Tanggal revisi: 4 Oktober 2026. Kontrak bersama: **m4human_kinetok_v3**. Status: rancangan penelitian dan development; bukan laporan training atau bukti keberhasilan. Arah ilmiah mengikuti [proposal terbaru](../proposal_riset_terbaru.md), input sensor mengikuti [01 — Encoder](01_encoder.md), target/readout mengikuti [03 — Decoder](03_decoder.md), alignment mengikuti [04 — LLM layer](04_llm_layer.md), dan pengujian mengikuti [05 — Evaluasi](05_evaluasi_testing.md).

## 1. Peran modul dan pertanyaan yang diuji

Penelitian mengembangkan metode sekaligus menguji hipotesis: apakah pembelajaran yang menjaga informasi kinematik **pada keluaran token akhir** membantu mempertahankan informasi gerak ketika representasi radar diringkas, dan apakah frozen LLM dapat memakainya pada tugas terpilih? Kompresi tidak diasumsikan pasti merusak informasi; kegunaan metode harus dibandingkan secara terukur.

Modul gabungan mempelajari representasi joint–waktu dan membentuk sejumlah kecil token kontinu. Nama file “dynamic model” dipertahankan untuk navigasi, tetapi modul ini bukan model transisi otonom, simulator biomekanik, atau forecaster. Ia memodelkan kinematika selama interval yang sudah diamati. Gaya, torsi, hukum Newton, kontak terukur, pusat massa, dan diagnosis keseimbangan bukan klaim utama.

Tiga hasil harus dibedakan:

- H/M: representasi kaya sebelum kompresi yang menjadi sumber bersama.
- U: token akhir yang benar-benar diberikan kepada projector; pelestarian informasi diuji di sini.
- Jawaban LLM: penggunaan token pada pertanyaan temporal/relasional berlabel terverifikasi.

Decoder H yang baik tidak membuktikan U baik. U yang mudah dibaca probe tidak otomatis mudah digunakan LLM. Kemenangan QA tidak membuktikan penguasaan seluruh dinamika fisik manusia.

### Penanda justifikasi

**F** = fakta sumber primer/audit yang disebutkan; **T** = identitas/derivasi dengan asumsi eksplisit; **R** = keputusan beralasan; **H** = hipotesis/default yang belum diuji. Nama model lokal, dimensi, kapasitas, loss weight, dan jadwal adalah R/H, bukan model open-source siap pakai atau konfigurasi optimal yang telah dibuktikan.

## 2. Pijakan literatur dan batas kebaruan

| Sumber primer | Fakta yang relevan | Batas inferensi |
|---|---|---|
| [RadarLLM](https://arxiv.org/html/2504.09862v2), metode dan Appendix H | Motion-guided Aggregate VQ-VAE menghasilkan token sekuens; language model dilatih. VQ meningkatkan ROUGE-L 29,8→31,2 pada radar-to-text-only. | Kelayakan radar-to-text, bukan bukti universal sensor understanding/frozen LLM atau bukti kompresi menghilangkan joint/velocity. Panjang token dan ukuran codebook berbeda. |
| [HuMoCon, CVPR 2025](https://arxiv.org/html/2505.20920v1) | Velocity reconstruction digunakan untuk detail perubahan gerak dan temporal over-smoothing. | Velocity supervision bukan gagasan pertama; ablation objective tidak mengisolasi jumlah token. |
| [FiGMo, preprint Juni 2026](https://arxiv.org/html/2606.20888v1) | Timestamp grounding relevan untuk QA temporal. | Pentingnya timestamp bukan bukti semua tokenizer terkompresi kehilangan urutan. |
| [SeMoCo, preprint Agustus 2026, §3.2/8.3](https://arxiv.org/html/2608.24334v2) | Decoder dari quantized tokens sudah disupervisi melalui rekonstruksi posisi serta velocity/acceleration consistency. | **Output-side kinematic loss sendiri bukan kebaruan.** Konteks sensor dan rangkaian pengujian kita harus menjadi pembeda yang terukur. |
| [MoTok, preprint Maret 2026, Appendix D.3](https://arxiv.org/html/2603.19227v1) | Selain generation, frozen tokenizer menyediakan tokens untuk captioner motion-to-text yang dilatih. | Compact-token-to-language juga memiliki pendahulu; jangan menggambarkannya sebagai generation-only atau bukti otomatis frozen-LLM QA. |
| [MotionBERT, ICCV 2023](https://arxiv.org/abs/2210.06551) | Representasi gerak memakai relasi spatial–temporal dan recovery motion 3D. | Preseden backbone, bukan checkpoint radar langsung kompatibel. |

**R — Kandidat kontribusi spesifik:** kombinasi konteks sensor-derived radar, final token dengan budget terbatas, pembandingan placement supervisi yang transparan, dan pembuktian physical readability serta penggunaan frozen LLM. Bukan “tokenizer pertama”, “dynamics pertama”, “velocity pertama”, “kinematic loss pada token pertama”, atau klaim RadarLLM pasti gagal. Attachment comparison adalah perlakuan yang diuji, **bukan bukti bahwa jenis loss-nya baru**. Kebaruan tetap memerlukan telaah literatur dan eksperimen; reproduksi seluruh model generatif tidak wajib untuk proof of concept ini.

## 3. Pipeline dan ownership training

~~~text
RPC terverifikasi → encoder E → predicted states/features
                                     ↓
                          common motion backbone
                                     ↓
                   H[B,32,23,128] → M_rich[B,32,23,256]
                                     ↓
                      chronological token learner C
                                     ↓
                              U[B,K,256]
                          ↙              ↘
                   probe fisik        projector → frozen LLM
~~~

| Tahap | Parameter yang dilatih | Dibekukan/deterministik |
|---|---|---|
| E — Persepsi | M4HumanSetEncoderV2 dan head posisi/root | Target recipe; tanpa LLM |
| M — Representasi bersama | CausalDSTformerLiteV1 + KinematicReadoutV2 pada H | Encoder terpilih |
| C — Kompresi | Token learner, fidelity decoder, auxiliary physical readout per kondisi | Encoder, motion, H/M/Z cache, recipe/labels |
| L — Alignment | Projector independen per kondisi/budget | Encoder, motion, token learner, semua physical heads, Qwen |

Motion backbone dipilih **sekali** pada train/validation lalu dibekukan sebelum membandingkan tokenizer. Tidak melatih ulang backbone hanya untuk C_kin. QA gradient tidak mencapai upstream; perubahan itu memerlukan protocol baru. Staged training tetap dapat disebut satu motion tokenizer pada tingkat fungsi sistem.

## 4. Data dan sensor input

Processed M4Human sekitar 50 GB berada pada perangkat lain dengan root rencana /dataset; schema/timestamps belum diaudit di sini. Ukuran disk bukan bukti kecukupan data. [Paper M4Human](https://arxiv.org/html/2512.12378v3), [repositori](https://github.com/FanJunqiao/M4Human), dan [reader/preprocessing](https://github.com/FanJunqiao/M4Human/blob/main/dataset/m4human_utils.py) adalah referensi eksternal, bukan audit paket pengguna.

Audit read-only LMDB wajib memastikan RPC XYZ/intensity, unit meter, calibration/frame alignment, timestamp detik, gap, urutan 22 joint, pelvis dan origin radar. Tidak mengasumsikan Doppler/SNR. Root adalah pelvis global; trans annotation tidak otomatis pelvis atau center of mass.

~~~text
P_enc_relative_m   [B,32,22,3]
r_enc_m            [B,32,3]
f_enc              [B,32,256]
time_s             [B,32]
joint_sensor_valid [B,32,22]
root_sensor_valid, feature_valid, context_time_valid [B,32]
~~~

Jika confidence joint belum tervalidasi, broadcast validity frame dan sebut dukungan frame. GT pose/root, annotation mask, body parameters, subject/action IDs, dan jawaban tidak masuk forward. Sensor validity dan target validity adalah jalur berbeda.

Encoder scratch reference memakai 512 point dan chronological context empat frame t−3..t, sesuai dokumen 01. Context tidak melintasi recording/gap/split; missing frame tidak diulang. L32 mencakup endpoint span 31/12≈2,58 detik pada grid nominal 12 Hz; timestamp aktual tetap authority.

Normalisasi input R/H: joint relatif dibagi 1 meter; root memakai mean/std train-only dengan floor std 0,001 m; f_enc memakai feature-only LayerNorm 256 tanpa affine. Simpan statistik, counts dan hash. NaN/Inf pada dukungan valid menghentikan run. Entri invalid disanitasi sebelum operasi, bukan dihapus melalui NaN×0.

Subjek held-out dipisahkan sebelum windows/cache/QA. Seluruh context/derivative support satu sampel tetap pada split sama. Stride awal train/val/test 8/16/16 adalah H; overlap bukan observasi independen.

## 5. Common backbone: CausalDSTformerLiteV1

### 5.1 Struktur dan transfer policy

Backbone lokal dilatih **scratch**. Yang diadaptasi dari [DSTformer sumber](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py) adalah pemisahan attention joint/time, bukan checkpoint/reproduksi exact MotionBERT. Input radar-predicted 3D, skeleton, width dan masks berbeda. Bobot MM-Fi/MotionBERT bukan drop-in. Point-MAE transfer hanya alternatif encoder lolos audit dokumen 01, bukan novelty utama.

~~~text
joint[t,j] = Linear(3,128)(P_input[t,j]) + joint_id[j]
global[t]  = Linear(259,128)(concat(r_input[t],f_input[t])) + joint_id[22]
X[B,32,23,128] = concat(joint,global) + fixed_time_PE128
~~~

Learned joint identity berukuran 23×128. Sinusoidal time encoding memakai x=(time_s−window_start)/(1/12 s). Encoding tersebut tidak mengganti timestamp nonuniform dengan waktu buatan.

**H — Empat stage** dengan dua branch berbobot independen:

~~~text
A(X) = X + Dropout(MHA(LN(X),LN(X),LN(X)))
branch_ST = temporal_ST(spatial_ST(X))
branch_TS = spatial_TS(temporal_TS(X))
Y = 0.5*branch_ST + 0.5*branch_TS
X_next = Y + Dropout(Linear(256,128)(GELU(Linear(128,256)(LN(Y)))))
H = final_feature_LN128(X_last)
~~~

Reference menggunakan empat attention heads, Linear dengan bias, GELU, LayerNorm eps 1e−5 dan dropout 0,1. Kedua branch sudah mempunyai residual; jangan menambahkan X lagi setelah mean. Semua temporal sublayer causal, tanpa temporal BatchNorm atau statistik seluruh window. Linear/MHA memakai Xavier-uniform dan bias nol, LayerNorm memakai weight satu, sedangkan joint IDs diinisialisasi normal dengan std 0,02. Seluruhnya default H, bukan optimum yang telah dibuktikan.

### 5.2 Mask dan causality

Spatial reshape [B×32,23,128]. Temporal permute dahulu menjadi [B×23,32,128], lalu inverse permute; direct reshape mencampur joint/time. Temporal mask future_key OR invalid_key. Guard all-blocked membuka dummy finite hanya untuk query invalid lalu zero output; valid query tidak membuka future.

Mask ulang setelah bias/LN/residual. Boolean True pada nn.MultiheadAttention berarti blocked; SDPA perlu konversi semantics eksplisit. [PyTorch MHA](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

Prefix-causality test pada backbone dan H physical head, eval/dropout nol, mengubah suffix setelah t. Ini availability test, bukan kausalitas fisik. Decoder U seluruh window berbeda kontrak pada bagian 8.

### 5.3 Representasi sebelum kompresi

~~~text
H            [B,32,23,128]
M_rich[t,j] = concat(H[t,j],H[t,22])       # termasuk j=22
M_rich       [B,32,23,256]
Z[t] = concat(masked_mean(H[t,0:22]),H[t,22])
Z            [B,32,256]
~~~

M_rich adalah **memory kaya yang benar-benar menjadi sumber kompresi**. Pada node global j=22, kedua bagian concat sama dan tidak menciptakan observasi tambahan. Z merupakan ringkasan diagnostik per frame yang dapat kehilangan informasi joint; ia bukan satu-satunya fidelity target dan bukan bypass ke LLM.

## 6. Training motion dan target fisik bersama

KinematicReadoutV2 dokumen 03 membaca H lewat dua head LN128→Linear128→GELU→Linear6. Output delta position dan velocity; posisi memakai residual predicted encoder. MPJPE rendah dapat dibantu input copy dan bukan bukti mandiri kandungan H.

~~~text
P_dec_relative = P_enc_relative + s_delta_joint*delta_joint
r_dec = r_enc + s_delta_root*delta_root
v_dec_relative = s_v_joint*output_v_joint
v_dec_root = s_v_root*output_v_root
L_M = L_position_joint + L_position_root
      + 0.5*(L_velocity_joint + L_velocity_root)/2
~~~

Target posisi memakai raw calibrated GT, sedangkan velocity memakai derivative yang difilter. Error posisi pada loss dibagi 1 m dan velocity memakai RMS train-only. Scale parameterisasi residual posisi disimpan terpisah dari scale error. Posisi/velocity pelvis relatif tepat nol dan dikeluarkan dari denominator yang mengukur joint nontrivial. Lambda acceleration, bone dan consistency tetap nol. Detail masks, scales dan empty groups mengikuti authority dokumen 03.

Default derivative polynomial endpoint causal tujuh sample, degree dua, actual timestamps. Enam endpoint pertama **setiap window** derivative-invalid walau cache recording punya history. Fit tau_i=time_i−time_t dan basis [1,tau,tau²], velocity=beta1. Pada uniform grid sesuai [SciPy endpoint coefficients](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html): window 7, polyorder 2, deriv 1, delta 1/12, pos 6, use dot. Duplicate/nonmonotonic timestamps, gap/rank-deficient fit ditolak; bukan centered filter dan tidak membagi dt lagi. Reference recipe aligned12hz_endpoint_sg_v2 menyimpan actual-time policy.

**T:** untuk independent position error variance sigma², variance first difference=2sigma²/dt². Correlation nyata mengubah hasil; mathematical filter dan AI dapat oversmooth. Comparator dari predicted pose/root yang sama, zero-velocity dan direct-rule tetap diagnostic wajib. Comparator pose-only tidak punya f_enc; bukan isolasi sempurna AI versus matematika.

## 7. Learned compressor: KinematicTokenLearnerV1

### 7.1 Budget dan bin kronologis

**R/H — Satu token kontinu per bin**, dengan width 256. K=16 adalah budget utama; K=8 dan K=32 menjadi budget sekunder yang ditentukan sebelum test. Reference tidak memakai radar vocabulary, VQ/codebook, duplikasi slot S/S, atau raw f_enc/pose bypass. Ketiga budget diperlukan untuk klaim multi-budget. Jika hanya K=16 feasible, laporkan hasil satu budget.

~~~text
start_k = floor(k*32/K)
end_k   = floor((k+1)*32/K)             # exclusive
K8: 4 frame/bin; K16: 2 frame/bin; K32: 1 frame/bin
U[B,K,256], token_valid[B,K]
~~~

Bin bounds memakai frame-time sensor. Bin time=midpoint timestamp endpoint bin relatif awal observasi; satu-frame bin=timestamp frame. Gap tidak diratakan. Common frame support = feature_valid AND root_sensor_valid AND all(joint_sensor_valid) AND context_time_valid. Policy sama antarcondition, bukan annotation mask.

Token valid jika bin mempunyai setidaknya satu common-valid frame. Empty bin U nol/mask false, bukan diam. Simpan original timestamps, bin bounds/counts, memory masks/source keys. Invalid entries bukan nol pengamatan.

### 7.2 Memory dan query attention

Reference memakai **satu learned query q0[256] shared antarbin**, sehingga kapasitas query tidak tumbuh ketika K berubah. Bin identity berasal dari fixed time encoding. Initializer q0 adalah Normal(mean=0, std=0,02), tanpa weight decay karena merupakan parameter satu dimensi. Seluruh initial state, termasuk q0, diklon dan checksum dicatat untuk pasangan kondisi; setiap budget memakai resep inisialisasi/seed yang sama, bukan checkpoint budget lain.

~~~text
M_raw[t,j]  = concat(H[t,j],H[t,22])       # M_rich/M pada cache
M_attn[t,j] = M_raw[t,j] + PE256(time_t)   # view sumber attention
memory_bin = masked_bin(M_attn)
query[k]    = q0 + PE256(bin_time_k)
Y[k] = query[k] + CrossAttention4(LN(query[k]),LN(memory_bin),LN(memory_bin))
U[k] = final_LN256(Y[k] + Linear(512,256)(GELU(Linear(256,512)(LN(Y[k])))))
~~~

Compressor menggunakan satu cross-attention, FFN ratio dua, dropout nol, Linear dengan bias dan LayerNorm eps 1e−5. Joint identity sudah tersedia di H; fixed time encoding dan provenance sama pada kedua kondisi. PE256 untuk memory, bin dan reconstruction query memakai x=(actual_time−window_start)/(1/12 s) dengan basis sinusoidal yang sama, bukan indeks bin yang dianggap detik. Satu helper `build_rich_attention_memory` membentuk M_attn dari M_raw dan menambahkan PE waktu **tepat sekali**. Compressor, auxiliary C_base, dan probe precompression memakai view serta mask yang sama; tidak ada tambahan joint embedding pada memory M_attn. Joint embedding decoder hanya pada query. M_raw tidak ditimpa saat caching; H tetap target fidelity tanpa PE tambahan. U akhir dipakai apa adanya pada decoder/projector, tanpa penambahan ulang encoding memory. H/M dibekukan dan compressor dilatih dari awal. Query tidak bergantung pada pertanyaan serta tidak membaca koordinat GT atau kategori jawaban.

Query hanya mengakses valid memory bin-nya. H frame bin dapat berisi history sebelumnya dari causal backbone. Tidak mengklaim token bersemantik joint tertentu atau tersedia sebelum endpoint bin. Empty-bin finite guard diikuti zero setelah bias/norm; all-empty scientific sample ditolak dengan reason.

**R:** learned weighting dapat memilih joint/time information sebelum mean-joint pooling membuang struktur. Chronological restriction memberi budget dan support terdefinisi. Itu hipotesis, bukan jaminan. C_pool dari masked pooling Z opsional diagnostic, bukan main baseline lemah.

## 8. C_base versus C_kin: attachment supervisi

### 8.1 Input dan full-H fidelity sama

| Unsur | C_base | C_kin |
|---|---|---|
| Encoder/motion/labels/split | Sama, frozen | Sama, frozen |
| Source | M_rich dari H sama | M_rich dari H sama |
| Compressor/K/width/init/budget | Sama | Sama |
| Common fidelity | U→seluruh fixed H | U→seluruh fixed H |
| Auxiliary physical head source | Fixed M_rich | U |
| Kinematic labels/scales/readout parameters | Sama | Sama |
| Kinematic gradient ke compressor | **Tidak** | **Ya** |

C_base **learned** dari full-H fidelity; bukan random atau mean baseline. Raw-label access/common physically trained upstream sama. Perlakuan yang disengaja ialah **lokasi supervisi kinematik**. Fixed M_rich membuat baseline auxiliary loss hanya melatih head; tidak berpura-pura gradient path identik atau pure architecture ablation.

### 8.2 Common full-H fidelity decoder

Training-only time+joint query decoder dari U:

~~~text
query[t,j] = fixed_time_PE256(t) + learned_joint_id256[j]
Q[B,32*23,256] --cross-attention4 + LN/FFN256→512→256--> hidden
H_hat_norm = Linear(256,128)(hidden), reshape[B,32,23,128]
H_target_norm = (stopgrad(H) - mu_H_train)/s_H_train
L_fidelity = masked_mean((H_hat_norm - H_target_norm)^2)
~~~

Decoder menghasilkan 128 nilai **dalam ruang H ternormalisasi** per query. mu_H_train adalah mean dan s_H_train adalah standard deviation per hidden channel, dihitung dari train dengan latent validity H AND common frame support yang sama dengan source tahap C. Floor std adalah 1e−3 (H); counts, kebijakan variance dan hash disimpan. Kedua kondisi memakai statistik yang sama. U invalid dimask. Target mask menggabungkan latent validity H dengan common frame support sehingga input support dan denominator sama. Rekonstruksi mencakup seluruh joint/global dan waktu eligible, bukan target mean-joint. Untuk visualisasi H asli, inverse transform adalah H_hat=mu_H_train+s_H_train*H_hat_norm. Kapasitas dan initial state readout sama. Z fidelity boleh menjadi diagnostik sekunder, **bukan pengganti full-H fidelity utama**.

Decoder merekonstruksi seluruh observed window **retrospective**: query t boleh membaca seluruh U window. Bukan online prediction saat t; tidak ada future di luar observasi. Time/joint query bukan coordinates/labels GT.

### 8.3 Auxiliary physical head

Kedua readout memakai arsitektur identik: width 256, cross-attention empat heads, time+joint query, FFN 512 dan final Linear dengan enam output. Ada 32×23 query: 22 joint relatif dan satu root global per waktu. Source adalah flattened M_attn[32×23,256] pada C_base, dibentuk dari M_rich melalui helper bagian 7, dan U[K,256] pada C_kin. Source mask menggabungkan latent validity dengan common frame support compressor; baseline tidak membuka source tambahan yang dilarang untuk U. Panjang source berbeda dicatat, tetapi jumlah parameter trainable sama.

~~~text
R_kin(source,time_s,joint_ids) -> position_relative/root + velocity_relative/root
L_kin = L_position_joint + L_position_root
        + 0.5*(L_velocity_joint + L_velocity_root)/2
L_Cbase = L_fidelity(U,H) + lambda_aux*L_kin(R_kin(M_attn),GT)
L_Ckin  = L_fidelity(U,H) + lambda_aux*L_kin(R_kin(U),GT)
lambda_aux = 1.0                         # H, locked on validation
~~~

Readout menghasilkan estimasi langsung **tanpa residual P_enc/r_enc, H skip, raw-feature bypass atau GT input**. Output/loss posisi dinormalisasi terhadap 1 m; velocity memakai scale train-only dari dokumen 03. Reference frame root, pelvis constraint, target dan support sama. Pada tahap C/probes, posisi valid jika endpoint common-frame-valid AND GT-valid. Velocity valid jika **seluruh tujuh frame support common-frame-valid** AND GT derivative-valid; enam endpoint awal invalid. Query tetap meliputi seluruh 32×23 posisi, tetapi loss/metrik hanya pada cohort eligible bersama. Mask tahap M H physical head dapat tetap per-joint; jangan menyamakannya dengan mask tahap C. P0 hanya memakai posisi dan velocity; task auxiliary loss tidak ditambahkan diam-diam. Jika kelak ditambahkan, kedua kondisi harus menerima label dan kontrol lokasi supervisi yang sepadan.

C_kin mendorong U membawa besaran terpilih. Sesudah freeze lakukan **probe baru**, bukan hanya menilai training head yang dioptimalkan bersama. Rekonstruksi tidak menjamin invertibilitas semua fisika atau language use.

## 9. Postfreeze probes dan attribution

Probe diagnostic baru dilatih train, dipilih validation, lalu panel held-out sama. Primary physical probe: time+joint query decoder kapasitas sama pada **M_rich versus U**; source lengths/counts dilaporkan. Z secondary. Raw-H spatial readout terpisah, bukan capacity-identical comparator.

Continuous metrics: joint/root position m/mm dan velocity m/s. Skor task-probe utama untuk root_speed_trend dan relative_limb_motion diperoleh dengan menerapkan recipe deterministik dokumen 03 pada prediksi posisi/velocity probe fisik independen setelah reliability gate. Task/body_part hanya menentukan agregasi recipe; classifier task langsung bukan bagian P0. limb_onset_order wajib sebelum klaim event order; jika onset labels tidak reliable, scope dipersempit sebelum test.

No-GT probe input/time-joint identity sama, no-state shortcut. Initializer/readout width/exposure/common source support/masks/cohort/recipe dikontrol. M probe tidak diberi frame yang dihapus dari source tokenizer. H decoder besar versus U linear probe kecil bukan bukti compression loss.

| Temuan | Interpretasi |
|---|---|
| M/probe gagal | Perception/label/representation bottleneck belum teratasi |
| M baik, U buruk pada controlled probe | Keterbatasan postcompression pada task/budget itu, setelah audit capacity/optimization |
| C_kin U lebih baik dari C_base U | Mendukung output-side supervision treatment pada konfigurasi teruji |
| U membaik, QA tidak | Retention belum menjadi language-use improvement |
| Baseline U sudah cukup baik | Dugaan bottleneck tidak didukung/terlalu kecil pada scope itu |
| Label/optimization/mask/runtime tidak valid | Engineering failure inconclusive, bukan penolakan hipotesis |

Caption/QA quality saja tidak menentukan informasi yang hilang. Effect praktis/toleransi dikunci dari validation sebelum test. Satu seed exploratory, bukan robustness lintas initialization.

## 10. Config dan urutan development

~~~yaml
contract_version: m4human_kinetok_v3
motion:
  id: CausalDSTformerLiteV1
  initialization: scratch
  length: 32
  spatial_nodes: 23
  width: 128
  stages: 4
  heads: 4
  causal: true
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.01
  max_epochs: 60
  micro_batch: 8
  accumulation: 4
compression:
  id: KinematicTokenLearnerV1
  condition: C_base_or_C_kin
  initialization: paired_identical_scratch
  rich_source: concat_H_joint_H_global
  input_dim: 256
  output_dim: 256
  primary_tokens: 16
  secondary_tokens: [8,32]
  shared_learned_query: true
  heads: 4
  layers: 1
  fidelity_target: full_fixed_H_32x23x128
  auxiliary_source: M_rich_for_C_base_U_for_C_kin
  lambda_aux: 1.0
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.01
  max_epochs: 30
  micro_batch: 2
  accumulation: 8
precision_reference: fp16_amp_with_grad_scaler
physical_loss_precision: fp32
seed: 42
~~~

Semua angka optimasi merupakan default H. Profil backward full-H decoder dilakukan sebelum menetapkan microbatch. AdamW memakai decay pada matrix weights, bukan bias, LayerNorm, atau q0. Warm-up fraction 0,05 diikuti cosine schedule; scheduler maju pada successful updates. Lakukan unscale sebelum gradient clipping. Pada tahap C, clip norm 1 secara **terpisah** untuk grup compressor+fidelity decoder dan grup auxiliary physical head, dengan grouping identik pada kedua kondisi. Jangan clip seluruh model sekaligus: besarnya gradien auxiliary C_base dapat menskalakan gradien fidelity compressor meskipun gradien auxiliary ke compressor nol. Pada C_kin, gradien auxiliary yang memang mencapai compressor tetap masuk grup pertama. Fixture finite harus membuktikan perubahan loss/gradien head auxiliary baseline tidak mengubah gradien maupun update compressor setelah clipping; numerical overflow adalah kegagalan run, bukan hasil ilmiah. FP16 dan GradScaler menjadi reference; BF16 hanya setelah capability dan numerical gate lulus. Simpan runtime dan precision aktual.

Urutan development: audit → synthetic tests → tiny-overfit pada 16–32 train windows → motion pilot dan mathematical comparator → pilih/bekukan common backbone → cache H/M/Z yang tetap → tiny-overfit kedua compressor → probes validation → training dengan matched budget → bekukan tokenizers → training projector independen pada dokumen 04. Test tidak dibuka untuk memilih task, lambda atau budget.

Run final C_base/C_kin memakai initial states yang diklon, sample/order/exposure sama, serta fixed successful-update budget dan validation schedule yang sama. Checkpoint dipilih melalui **minimum normalized full-H fidelity loss pada validation**, dengan tie-break checkpoint lebih awal. Physical metrics menjadi diagnostik terpisah, bukan selection objective campuran yang berubah antarcondition. Cap jadwal sama; tidak ada early stop atau extra fine-tuning sepihak pada run final. Pilot boleh mengubah defaults bersama sebelum test dikunci. Laporkan loss tanpa bobot, masks, jalur gradien, jumlah parameter aktual dan biaya.

## 11. Resource, cache dan reproducibility

Target perangkat adalah satu RTX 3060 dengan VRAM 12 GB dan RAM 16 GB. Implementasi memakai Pure-PyTorch serta .venv/uv. Belum ada benchmark perangkat atau jaminan waktu/VRAM. Encoder dan LLM tidak perlu resident pada GPU saat tahap C.

**T — Ukuran per unique window fp16:** H=32×23×128×2=188416 byte atau 184 KiB. M_rich jika disimpan memerlukan 368 KiB; Z memerlukan 16 KiB; U pada K=8/16/32 masing-masing 4/8/16 KiB. M_rich dapat dibentuk dari shard H saat dibaca. Full-H decoder mempunyai 736 query; backward perlu diprofil, bukan dianggap murah hanya karena token sedikit.

H diperlukan untuk training compression **dan** probes, bukan hanya debug. Gunakan bounded recording/shard arrays, unique-window cache dan lazy loading. Audit ukuran disk serta CPU RSS; jangan preload 50 GB atau seluruh H/M dan jangan menggandakan cache per paraphrase. Artefak turunan ditulis di luar LMDB sumber. Cache Z dan physical predictions bersifat opsional sesuai kebutuhan.

Bundle: window/frame/time/context/masks; source/split/encoder/motion/readout hashes; H dtype/layout; normalizers/joint map/derivative recipe; budget/bin policy dan runtime. U berasal frozen tokenizer yang sama dengan alignment/inference. Upstream berubah berarti cache baru. Tidak cache projector output selama training.

Strict load menolak unknown ID/shape/recipe/foreign hashes. Resume optimizer/scaler/scheduler/RNG/sampler/successful-skipped updates; epoch-boundary default. Catat GPU allocated/reserved peak, CPU RSS, median/p95 step, windows/sec, I/O. Report run baru setelah training nyata; tidak menimpa artefak MM-Fi.

Logging M/C mengikuti [dokumen 05 §12.2–12.3](05_evaluasi_testing.md#122-kontrak-log-minimum-untuk-analisis-pascarun): term p/v dan full-H fidelity/auxiliary dipisahkan, counts/support serta norm gradient per grup dicatat, selected-checkpoint records dan panel trajectories disimpan. Diagnosis membedakan upstream, kompresi dan probe; iterasi K16 melalui train/validation memakai parent/config diff dan fairness matched, tanpa menambah scope P0 atau menjamin gain.

## 12. API rencana dan gates

API berikut belum dibuat oleh revisi dokumentasi ini:

~~~text
motion.forward(sensor_state,sensor_masks,time_s) -> H,Z,latent_masks
make_rich_memory(H,time_s,masks) -> M_rich,rich_masks
tokenizer.forward(M_rich,time_s,masks,K) -> U,token_mask,bin_provenance
fidelity_decoder.forward(U,time_queries,joint_queries) -> H_hat_norm
physical_readout.forward(source,time_queries,joint_queries) -> p,v
compression_loss(H_hat_norm,fixed_H,H_normalizer,physical_prediction,target_sample) -> groups,coverage
export(frozen_bundle,window_manifest) -> versioned_H_or_U_cache
~~~

TargetSample hanya dibaca loss/evaluator, bukan motion/tokenizer/projector forward. Kode MM-Fi yang sudah ada dapat menjadi referensi penggunaan ulang, bukan bukti implementasi M4Human selesai.

| Gate | Check minimum |
|---|---|
| Data/target | Unit/calibration/root/map/time/split; annotation changes tidak mengubah U |
| Derivative | Constant/linear/quadratic/irregular-time fixtures; first-six invalid; no gap |
| Backbone | Permute round-trip/prefix causality/padding invariance/finite invalid guard |
| Tokenizer | K8/16/32 exact bins/masks; no question dependence; empty zero; no bypass |
| Fairness | Same H/M/labels/init; full-H fidelity; auxiliary source/grad attachment benar |
| Autograd tahap C | C_base auxgrad ke compressor nol/fidelity ada; C_kin auxgrad ada; upstream checksum tetap |
| Readouts | Whole-window retrospective; time/joint queries sah, GT coordinates/kategori jawaban dilarang; units/masks/counts benar |
| Proof | Postfreeze matched M/U probes+task reliability; bukan training-head score |
| Resource/reload | Tiny-batch full-H backward; strict bundle/cache-live equivalence; bounded RAM |

Lulus gerbang berarti studi layak diuji, bukan kebaruan atau keunggulan telah terbukti. Hasil negatif bermakna jika data, optimasi dan uncertainty memadai. Lanjut [03 — Decoder](03_decoder.md) untuk label/probe dan [04 — LLM layer](04_llm_layer.md) untuk memastikan hanya U diberikan ke frozen LLM.
