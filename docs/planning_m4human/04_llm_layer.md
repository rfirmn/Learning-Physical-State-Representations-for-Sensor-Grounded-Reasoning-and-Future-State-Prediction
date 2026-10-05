# 04 — Token Akhir, Projector-Only Alignment, dan Frozen LLM

Tanggal revisi: 4 Oktober 2026. Kontrak: **m4human_kinetok_v3**. Status: rencana development/evaluasi, bukan hasil training, akses dataset atau kelayakan GPU yang sudah diverifikasi. [Proposal](../proposal_riset_terbaru.md) menetapkan pertanyaan/kebaruan; [01](01_encoder.md) input sensor; [02](02_dynamic_model.md) motion/tokenizer; [03](03_decoder.md) targets/evidence/readouts; [05](05_evaluasi_testing.md) pengujian dan batas klaim.

## 1. Tujuan lapisan bahasa

Lapisan ini menguji apakah **token akhir U yang sudah dibekukan** dapat digunakan frozen LLM untuk menjawab pertanyaan kinematik terpilih. Ia bukan tempat menciptakan kembali informasi yang hilang selama kompresi. Hasil QA dilaporkan bersama keterbacaan fisik U, bukan menggantikannya.

Perbandingan utama adalah **C_base versus C_kin**, dengan 16 token pada kedua kondisi. C_base mempelajari token melalui rekonstruksi seluruh H; C_kin menggunakan mekanisme yang sama dengan tambahan supervisi kinematik yang membaca U. Keduanya memakai encoder, motion backbone, memory kaya, label, arsitektur dan inisialisasi compressor, serta target fidelity yang sama. Perbedaan lokasi supervisi dan jalur gradien dijelaskan di dokumen 02. Baseline tidak boleh digambarkan sebagai sistem tanpa motion atau tanpa supervisi fisik.

**F/T/R/H** masing-masing berarti fakta sumber primer/audit, derivasi berasumsi, keputusan beralasan, dan default/hipotesis belum diuji. Semua nama implementasi lokal di dokumen ini adalah rancangan, bukan library open-source yang telah tersedia.

### Batas kontribusi

[RadarLLM](https://arxiv.org/html/2504.09862v2) mendukung kelayakan radar-to-text dengan model bahasa yang dilatih, bukan jaminan frozen LLM memahami seluruh sensor. [SeMoCo, §3.2/8.3](https://arxiv.org/html/2608.24334v2) sudah menerapkan supervisi posisi dan velocity/acceleration consistency pada decoder quantized tokens. [MoTok, Appendix D.3](https://arxiv.org/html/2603.19227v1) juga memakai frozen motion tokenizer untuk melatih captioner motion-to-text. Karena itu **output-side kinematic loss dan token-to-language sendiri bukan kebaruan**. Kandidat kontribusi adalah kombinasi konteks radar sensor-derived, token akhir dengan budget terbatas, physical readability, dan penggunaan frozen LLM yang diuji melalui kontrol sepadan. Jenis loss atau perubahan dataset saja tidak menjamin kontribusi ilmiah.

Pertanyaan terstruktur dapat dipecahkan sebagai pemilihan kategori. Keberhasilan menunjukkan sistem memakai bukti pada task yang diuji, bukan reasoning universal, penguasaan hukum Newton, niat manusia, gaya/torsi atau diagnosis keseimbangan. Forecasting bukan objective utama.

## 2. Kontrak upstream dan masukan yang benar-benar diterima LLM

### 2.1 Dataset dan lineage

Paket processed M4Human berada pada perangkat eksperimen lain, dengan root rencana /dataset. Ukuran sekitar 50 GB menunjukkan kebutuhan disk, bukan jumlah observasi independen atau jaminan kecukupan training. Reader harus membuka LMDB secara read-only. Audit schema, unit, kalibrasi, timestamp, gap, mapping 22 joint dan pelvis wajib dilakukan sebelum membentuk cache atau QA. Kanal RPC XYZ/intensity digunakan hanya setelah verifikasi; Doppler dan SNR tidak diasumsikan tersedia. [Repositori resmi M4Human](https://github.com/FanJunqiao/M4Human).

GT, body parameters, annotation masks dan action IDs hanya digunakan pada jalur target atau evaluasi. Inferensi menerima sensor, kalibrasi/waktu yang sah dan pertanyaan. Mengubah GT tidak boleh mengubah U atau prefix dari window sensor yang sama. Field trans tidak otomatis menyatakan pelvis; root juga bukan pusat massa. Kontrak MM-Fi dengan 17 joint, latent 384 dimensi dan forecasting B3–B4 tetap artefak historis. Bobot dan hasilnya tidak diklaim sebagai hasil M4Human.

### 2.2 Tahap freeze dan tensor

| Tahap/tensor | Kontrak |
|---|---|
| Encoder | M4HumanSetEncoderV2 dilatih dari awal; 512 point, RPC empat kanal dan causal context empat frame setelah audit; fitur 256 dimensi serta prediksi 22 joint relatif dan pelvis global |
| Motion bersama | CausalDSTformerLiteV1 dilatih dari awal; L=32, 23 node, width 128, empat stage; physical head membaca H |
| H | [B,32,23,128], hidden joint–waktu yang dibekukan setelah tahap M |
| M_rich | [B,32,23,256], concat(H_tj,H_tglobal); memory kaya yang benar-benar menjadi sumber kompresi |
| Z | [B,32,256], mean joint dan global; representasi diagnostik sekunder |
| Tokenizer | KinematicTokenLearnerV1 menggunakan cross-attention pada bin kronologis; arsitektur sama untuk kedua kondisi |
| U | **[B,K,256]**, satu token kontinu per bin; mask, waktu bin dan provenance berasal dari sensor |
| Projected tokens P | [B,K,d_llm], satu-satunya physical embeddings yang masuk LLM |

Urutan pelatihan adalah: encoder dilatih lalu dibekukan pada tahap E; common motion backbone dan readout H dilatih lalu dibekukan pada tahap M; compressor dan training heads dilatih dari cache H/M yang tetap pada tahap C; terakhir, **hanya projector** dilatih pada tahap L. Pada alignment, seluruh upstream dan physical heads dibekukan. Loss QA tidak melatih tokenizer, motion backbone atau encoder. Protocol utama tidak menggunakan LoRA maupun end-to-end fine-tuning.

K=16 adalah budget utama; K=8 dan K=32 merupakan budget sekunder yang ditentukan sebelum test. Pada setiap budget, kedua kondisi memakai K, width, mask bin dan dukungan window yang sama. Setiap bin berisi 32/K frame. Validitas token ditentukan dari dukungan sensor bersama, bukan ketersediaan target. Bin kosong menghasilkan nol dengan mask false. Encoding waktu dan provenance mengikuti dokumen 02; projector tidak menambahkan resampler yang dilatih atau slot tambahan.

**Tidak ada raw f_enc, predicted pose/root, full H/M/Z, decoder numerical facts atau GT bypass di main LLM input.** Physical decoder/readout dipakai training/probe/evidence terpisah, bukan jalur tersembunyi yang menjawab pertanyaan untuk LLM. Jika explicit-facts diagnostic ditambahkan, identitas kondisi dan biayanya terpisah.

### 2.3 Cache U dan actual information test

U harus berasal frozen tokenizer checkpoint yang sama dengan alignment/inference. Cache per unique window, tidak per pertanyaan/paraphrase. Simpan window/frame/time/context/masks/bin support, source/split/map/recipe/encoder/motion/tokenizer hashes dan K. Cache output projector P tidak digunakan selama projector berubah.

Forward H/M mengikuti window L=32 dengan kebijakan context dan reset posisi yang sama. Forward satu recording penuh tidak otomatis ekuivalen. Dukungan diputus pada batas recording, gap dan split. Sanitasi entri invalid dilakukan sebelum operasi; NaN/Inf pada dukungan valid dilaporkan sebagai kegagalan dengan error row.

M pada cache ialah concat H mentah (M_raw). Sebelum attention tahap C, helper bersama menambahkan PE waktu sensor tepat sekali menjadi M_attn, dengan source mask yang sama bagi compressor, auxiliary baseline, dan probe M. Detail initializer/query dan clipping mengikuti dokumen 02/03. U akhir sudah mengandung hasil encoding ini: readout dan projector tidak menambahkan ulang encoding memory atau membuka full M sebagai jalur input bahasa.

Sesudah tokenizer dibekukan, jalankan **physical probes M_rich versus U** dengan kapasitas dan dukungan sumber yang dikontrol, sesuai dokumen 03/05. Z menjadi diagnostik sekunder. Readout yang dilatih bersama tokenizer bukan satu-satunya bukti pelestarian informasi. Jika probe yang layak tidak dapat membaca bukti tugas dari U, memperbesar alignment tidak otomatis menyelesaikannya. Namun kapasitas dan optimasi probe tetap diaudit sebelum menyimpulkan informasi tidak tersedia.

## 3. Pembanding ilmiah dan kesetaraan alignment

| Kondisi | Input bahasa | Fungsi |
|---|---|---|
| C_base | U_base pada K fixed + projector sendiri | Full-H-fidelity learned tokenizer control |
| C_kin | U_kin pada K sama + projector sendiri | Output-side kinematic-supervision treatment |
| C_text | Pertanyaan/schema sama, tanpa sensor tokens | Diagnostic prior bahasa/format |
| C_majority | Majority train per task; tanpa LLM | Baseline murah |
| C_rule | Mathematical/predicted-state evidence; tanpa LLM | Physical task feasibility diagnostic |
| C_probe | Probe fisik independen pada frozen M/U; jawaban task melalui recipe deterministik, tanpa LLM | Retention/extractability diagnostic; classifier task langsung bukan bagian P0 |

Pada tahap C, kedua kondisi memakai full-H fidelity **U→H[B,32,23,128]**, target fisik dan jumlah parameter readout yang sama. Auxiliary readout C_base membaca M_rich yang tetap sehingga loss kinematik tidak memberi gradien ke compressor. Readout C_kin membaca U sehingga gradien tersebut mencapai compressor. C_base tetap mempelajari compressor melalui full-H fidelity. Ini pengujian lokasi supervisi, bukan ablation arsitektur murni atau klaim semua jalur gradien identik.

**Matched alignment R:** projector architecture/initial state, Qwen checkpoint, tokenizer/runtime, prompts, QA IDs/order/sampler/exposure, token count/masks/intervals, precision, update budget, validation schedule, parsing/generation sama. Kedua projector di-reset ke state awal bersama dan dilatih independen; jangan memakai projector yang dilatih C_kin lalu mengujinya pada C_base sebagai comparison utama.

Klaim multi-budget memerlukan pasangan run dan probe yang sepadan pada K=8 dan K=32. Budget pemenang tidak dipilih dari test; K=16 tetap primary. Jika sumber daya hanya memungkinkan K=16, laporkan hasil satu budget dan batasi klaim.

C_pool, C_state, explicit predicted facts, static-history, multi-seed atau cross-action hanya diagnostic/claim-dependent. Comparison lama C_history/C_dynamics bukan primary karena berbeda akses state, parameters dan supervision. Tidak menambah seluruh baselines generatif hanya untuk memenuhi daftar panjang.

## 4. Model bahasa dan transfer policy

**F — Referensi:** Qwen/Qwen2.5-1.5B-Instruct pretrained/post-trained, bukan scratch. [Model card resmi](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [config resmi](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json).

| Field resmi | Referensi |
|---|---:|
| Architecture | Qwen2ForCausalLM |
| Hidden size |1536|
| Layers |28|
| Query/KV heads |12/2|
| Intermediate size |8960|
| Vocabulary |151936|
| Tied embedding/lm_head |true|
| Attention dropout |0,0|

Projector membaca hidden_size aktual dari model.config. Dtype bfloat16 pada asset bukan bukti capability atau kelayakan numerik pada RTX 3060. Bahasa Indonesia/JSON/task kinematik perlu pilot, bukan diasumsikan dari model card.

Load pretrained full weights dengan safetensors, trust_remote_code=false, model/tokenizer revision full commit yang sama, satu device CUDA eksplisit. Simpan config/chat-template/generation hashes. Semua embedding/attention/MLP/norm/lm_head **requires_grad=false**. Tidak LoRA, vocabulary resize, learned language backbone atau weight update LLM.

Pin runtime PyTorch/Transformers saat implementation. Dokumentasi Transformers 4.45.2 dipakai sebagai API reference yang bisa ditelusuri, bukan klaim latest/installed. Upgrade harus mengulang gate. Pure-PyTorch dan .venv/uv mengikuti AGENTS; tanpa external CUDA/FlashAttention extension atau offload/quantization tambahan otomatis.

Projector learned menguji penghubung distribusi sensor–language. Feature alignment berpreseden pada [LLaVA training](https://github.com/haotian-liu/LLaVA#train), tetapi full protocol karya itu bukan protocol frozen-Qwen radar ini. Encoder/tokenizer tidak pretrained otomatis hanya karena memakai nama inspired.

## 5. QA dan target yang bisa dipertanggungjawabkan

### 5.1 Satu task registry

Authority definisi task/evidence adalah dokumen 03: satu task_definitions.json/label_recipe/derivative recipe berversi dan hashes yang dipakai QA generator, prompts, probes, parser, evaluator. Dokumen 04 tidak menciptakan thresholds/enum kedua.

Untuk `root_speed_trend`, template mengikuti pertanyaan kanonis dokumen 03 tentang tren speed **secara keseluruhan**. `speeding_up`/`slowing_down` tidak berarti monoton setiap frame; `no_overall_trend` menyatakan slope dalam deadband pada support/fit sah, bukan speed selalu konstan. `unknown` tetap untuk support atau kualitas fit yang tidak memadai. Generator QA, jawaban canonical dan parser mengikuti enum registry yang sama.

Minimum primary tasks **root_speed_trend** dan **relative_limb_motion**, jika keduanya lolos reliability/support/class-coverage gate. Event-order task **limb_onset_order** hanya masuk setelah onset/quiet-baseline/confirmation/censoring reliability independen. Tanpa itu proposal tidak mengklaim pelestarian event order. root_radial_direction opsional setelah origin/root/reference-frame gate.

Interval tugas berbasis derivative dimulai pada time_s[6] sampai time_s[31]; interval observasi adalah time_s[0] sampai time_s[31]. Pada grid nominal 12 Hz, tugas dimulai 0,5 detik setelah awal observasi dan endpoint sekitar 2,583 detik. Waktu dan support aktual tetap dicatat. Jawaban tidak mengklaim velocity valid pada enam frame awal. Context encoder tiga frame sebelum window dicatat dan sama pada kedua kondisi; target derivative tidak membaca GT sebelum interval observasi yang dideklarasikan.

Questions menyebut body part, interval dan reference frame yang perlu. Root-relative limb motion berbeda dari global velocity. Jawaban kategori tidak berarti model mengetahui sebab/gaya/niat gerak. Template wording bukan observasi sensor baru.

### 5.2 GT recipe dan status target

GT mentah tidak diubah. Posisi menggunakan koordinat terkalibrasi; velocity diturunkan melalui polynomial endpoint causal dengan tujuh sample, degree dua dan timestamp aktual. Enam endpoint awal invalid. Recipe kinematic_recipe_v3 dan aligned12hz_endpoint_sg_v2 menyimpan hash serta kebijakan support yang digunakan. Grid nominal bukan pengukuran 100 Hz; interpolasi tidak menciptakan MoCap baru. Detail derivative, unit dan masks mengikuti dokumen 03, sama dengan comparator direct-rule. Pada tahap C dan probes, posisi memerlukan common frame support pada endpoint serta GT valid; velocity memerlukan common frame support pada seluruh tujuh sample dan derivative GT yang valid. Mask H physical head tahap M dapat berbeda karena per-joint; perbedaan itu tidak boleh memperkaya source baseline pada tahap C.

| Status | Perlakuan |
|---|---|
| defined / valid | Target kategori reliable; masuk panel eligible |
| unknown / unknown | Ambiguity/unknown yang dapat diaudit sesuai registry; bukan otomatis GT missing |
| undefined / excluded | Target tidak terdefinisi/tidak applicable/anotasi gagal audit; reason/coverage tersimpan |
| Model unknown pada target defined | Abstention salah pada strict score; dilaporkan terpisah |
| Parse/runtime failure pada eligible QA | Salah; tidak dihapus dari denominator |

Mapping target_status dan validity one-to-one di registry; tidak boleh berubah bebas. Evidence undefined berbeda dari task target unknown/exclusion. Unknown bukan confidence terkalibrasi. Model accuracy tidak menentukan label eligibility. Raw LMDB tidak ditimpa; perubahan recipe menghasilkan derived artefact baru.

### 5.3 Manifest, sampling dan leakage

Manifest: qa_id/window_id/recording_id/subject group, frame/source keys,time_s,observation/task intervals,task/body part/question/template,answer/target_status/validity/evidence_status/reason,label-support,GT/recipe/map/split hashes,upstream/input-cache lineage. IDs/GT metadata hanya join/evaluator; bukan prompt hints.

~~~text
make_qa(GT_evidence,registry,templates,split_manifest) -> records + coverage
join_input(window_id,frozen_U_cache) -> U,token_mask,provenance
build_prefix(question,task_definition,U) -> prefix tanpa target/evidence GT
~~~

Split subjek sebelum window/QA, seluruh context/support/paraphrase tetap pada split sama. Train balancing dicatat; val/test panel fixed, tidak dipilih yang mudah. Hitung unique windows/recordings/subjects, task/class support, unknown/exclusions. Batas pilot 1.000–3.000 QA adalah default H jika data yang sah tersedia. Jumlah final ditentukan dari audit, biaya dan validation, bukan janji kecukupan data.

Main answers JSON canonical pendek, satu task/answer sesuai registry, stdlib serialization deterministic. Free-form explanation bukan metric utama. Questions tidak memuat kategori jawaban, GT confidence/action labels, file path, subject ID atau body parameters.

## 6. Projector dari token aktual U menuju P

**H — M4HumanPhysicalProjectorV1**, scratch satu per condition/budget/seed:

~~~text
U[B,K,256]
 → LayerNorm256(affine)
 → Linear256→512(bias)
 → GELU
 → Linear512→d_llm(bias)
 → alpha fixed buffer
 → P[B,K,d_llm]
~~~

Projector dibagi antarbin. Reference tidak memakai output LayerNorm, dropout, resampler, learned positional embedding atau parameter output scale. U sudah membawa representasi bin kronologis dari dokumen 02. LayerNorm mengontrol skala fitur; MLP menyediakan pemetaan nonlinear kecil. Keduanya keputusan R/H, bukan optimum yang telah dibuktikan.

**T — Untuk d_llm=1536:** LayerNorm memiliki 512 parameter, Linear pertama 131.584 dan Linear kedua 787.968; total **920.064**. Rumus umum adalah 132096+513×d_llm. Verifikasi numel aktual dan strict state_dict. Projector legacy dengan susunan berbeda bukan model ini. Linear memakai Xavier-uniform dan bias nol; LayerNorm memakai weight satu dan bias nol. Simpan initial state bersama yang diklon untuk paired conditions.

Alpha buffer tetap dikalibrasi satu kali dengan ≤64 unique train windows C_base-K16 tanpa jawaban, initial projector bersama:

~~~text
alpha = median(norm(text_prefix_embeddings)) /
        max(median(norm(unscaled_projector(U_valid))),1e-8)
~~~

Cohort, template, initial state dan alpha sama untuk kedua kondisi serta budget sekunder; simpan diagnostik norm. Rentang 1e−3 sampai 1e3 adalah H untuk mendeteksi preprocessing yang degenerate, bukan batas silent clamp. Alpha tidak dikalibrasi ulang setiap epoch atau dari test, dan bukan parameter yang dilatih. Perubahan policy dikunci bersama sebelum test.

## 7. Prefix embeddings, loss masks dan generation

### 7.1 Boundary yang identik training/inference

Qwen2 API mendukung inputs_embeds/mask/position_ids. [Reference API](https://huggingface.co/docs/transformers/v4.45.2/model_doc/qwen2), [chat templating](https://huggingface.co/docs/transformers/v4.45.2/chat_templating).

1. System instruksi JSON; user marker internal unik di baris tersendiri lalu question/interval/body part. Input question dilarang menyisipkan marker/control tokens.
2. Render template sampai assistant header dengan add_generation_prompt=true. Marker harus tepat sekali.
3. Split text_pre/text_post, tokenize dengan add_special_tokens=false; sisipkan P pada boundary marker. No radar vocabulary IDs.
4. Training append canonical answer IDs+assistant terminator dari template yang dipin; prefix identik inferensi. Jangan joint-retokenize prefix+answer sehingga BPE boundary berubah.
5. Golden tests: target berubah hanya setelah assistant header; prefix IDs/embeddings/masks/positions tetap.

~~~text
[text_pre embeddings] + [P physical embeddings] + [text_post question/header]
training saja: + [answer JSON embeddings] + [assistant terminator]
~~~

| Bagian | Labels | Attention |
|---|---|---|
| System/user/question/assistant header |−100|1|
| Physical valid |−100|1|
| Physical invalid |−100|0|
| Answer+terminator |actual token IDs|1|
| Padding |−100|0|

Assemble sample lengkap lalu right-pad training/left-pad generation; jangan pad text fragments sendiri. Position IDs forward dari cumulative valid attention, masked positions consistent. Actual bin time tetap ditangani representation/provenance, tidak diganti index token sebagai physical timestamp.

Causal labels shift **sekali**. Untuk per-QA loss, helper membandingkan logits[:,:−1] dengan labels[:,1:], memilih non−100, mean per QA lalu mean antar-QA. Tidak memakai outputs.loss global token-mean seolah-olah per-QA mean. Single-QA fixture harus cocok dengan library loss. Jika library labels-loss digunakan langsung pada variant, labels tetap unshifted; jangan double shift. [Qwen2 source 4.45.2](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py).

### 7.2 Length dan deterministic generation

**H:** panjang total training S≤256 mencakup K token fisik, chat, jawaban dan terminator. Prefix generation ≤192 dengan max_new_tokens=64. Audit panjang maksimum, termasuk K=32, dilakukan sebelum panel dikunci. Sample terlalu panjang ditolak dengan reason/count atau template dipendekkan secara deterministik pada train/validation. Sensor dan jawaban tidak dipotong diam-diam. Sampel eligible pada test terkunci yang ditolak runtime tetap memiliki failure row, bukan dikeluarkan ulang secara sepihak.

Generation menggunakan greedy decoding: do_sample=false, num_beams=1, max_new_tokens=64, repetition_penalty=1 dan use_cache=true. EOS, pad dan generation IDs berasal dari assets yang dipin. Temperature/top_p tidak digunakan untuk sampling. Cache baru dibuat untuk setiap request; KV cache tidak dibagi lintas QA.

Returned IDs pada embeddings-only path bisa berbeda dari input_ids generation; regression test versi runtime menentukan bagian decode. Simpan raw IDs/text. Compare raw first-answer logits teacher-forced versus generate output_logits, bukan processed scores, precision dan backend yang sama; fixture argmax bermargin jelas dan tolerance awal 1e−2 H. Individual/batched padding setara; runtime upgrade ulang gate. [Generation API reference](https://huggingface.co/docs/transformers/v4.45.2/internal/generation_utils).

## 8. Autograd, objective dan precision

| Modul tahap L | Update |
|---|---|
| Encoder/common motion |Tidak, eval|
| Tokenizer/fidelity+physical heads |Tidak, eval|
| Recipe/alpha |Deterministik/buffer|
| Projector |**Ya**|
| Qwen termasuk embedding/lm_head |Tidak; forward training tetap grad-enabled|

**T:**

~~~text
L_QA = mean_QA[-mean_answer_token log p_frozenQwen(y_m | P(U),q,y_<m)]
grad_projector = J_projector^T * grad_P L_QA
~~~

Frozen berbeda dari no_grad/eval. Requires_grad=false pada bobot Qwen membatasi update, bukan meniadakan derivative terhadap P. Jangan no_grad/inference_mode pada **LLM training forward**. Cast P ke dtype LLM tetap differentiable; no detach. U dibaca sebagai ordinary detached cache tensor; inference_mode-created tensors tidak dipakai langsung jika backward perlu menyimpannya. [PyTorch autograd notes](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/docs/source/notes/autograd.rst).

~~~text
U = ordinary_detached_cache_tensor
P = projector(U)
embeds = concatenate(frozen_text_embeddings,P_cast,answer_embeddings)
logits = llm(inputs_embeds=embeds,attention_mask=mask,
             position_ids=positions,use_cache=False).logits
loss = per_QA_answer_only_causal_CE(logits,labels)
backward(loss)                       # optimizer projector saja
~~~

Smoke test menyimpan gradien P dan memastikan gradien kedua Linear finite serta nonzero, parameter projector berubah, sedangkan frozen gradients tetap None dan checksum tidak berubah. Prefix tekstual maupun target GT tidak mempunyai jalur sensor trainable tersembunyi.

**H — Precision:** reference memakai FP16 autocast dan GradScaler; master parameters dan optimizer projector memakai fp32. BF16 hanya digunakan setelah capability dan numerical gates forward/backward/generation lulus. Target fisik, export dan probe memakai fp32 sesuai dokumen 03. Catat dtype, scaler dan backend aktual; policy diagnostic/fallback FP32 juga dicatat. [PyTorch AMP](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/docs/source/amp.rst).

Training memakai use_cache=false. Aktivasi untuk backward terhadap input tetap diperlukan walaupun bobot LLM dibekukan. Eval/generation memakai no_grad atau inference_mode. Native checkpointing bersifat opsional setelah profiling: reference source Qwen 4.45.2 memerlukan flag aktif **dan model.training**; flag pada mode eval saja tidak cukup. Bila dibutuhkan, verifikasi Qwen frozen dalam train mode dengan dropout nol dan non-reentrant checkpointing. Hook/counter harus membuktikan recomputation, lalu model kembali ke eval untuk generation. Jangan memanggil wrapper.train secara buta sehingga upstream berubah menjadi stochastic.

## 9. Training matched, selection dan resume

~~~yaml
contract_version: m4human_kinetok_v3
condition: C_base_or_C_kin
token_budget: 16
upstream: frozen_encoder_motion_tokenizer_and_all_readouts
llm:
  model: Qwen/Qwen2.5-1.5B-Instruct
  revision: full_commit_at_setup
  freeze: true
  lora: false
projector:
  id: M4HumanPhysicalProjectorV1
  input_dim: 256
  hidden_dim: 512
  output_dim: model_config
  initialization: paired_identical_scratch
  alpha: common_train_only_buffer
training:
  micro_batch: 1
  accumulation: 16
  max_total_tokens: 256
  max_prefix_tokens: 192
  epochs_cap: 5
  optimizer: AdamW
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_fraction: 0.05
  clip_grad_norm: 1.0
  seed: 42
precision_reference: fp16_amp_with_grad_scaler
generation:
  do_sample: false
  beams: 1
  max_new_tokens: 64
~~~

Semua angka optimization adalah default H. AdamW memakai betas (0,9; 0,999) dan eps 1e−8; weight decay hanya pada matrix weights, bukan bias atau LayerNorm. Warm-up/cosine mengikuti successful optimizer updates. Overflow dan skipped updates dicatat. Loss dirata-ratakan per answer token pada setiap QA, kemudian antar-QA dalam accumulation group aktual. Group terakhir tidak dibagi 16 jika jumlah QA kurang dari itu. Accumulation tidak mengurangi peak activation satu microbatch.

1. Jalankan synthetic smoke untuk shape, mask, gradient, generation dan profil sumber daya pada perangkat eksperimen.
2. Lakukan pilot train/validation, tiny-overfit, audit format dan probe U. Test tidak dipakai untuk memperbaiki atau memilih model.
3. Reset kedua projector ke initial state bersama. Gunakan urutan QA, exposure, successful-update budget dan validation schedule yang sama. Pilih checkpoint masing-masing melalui aturan generated validation yang sama.
4. Kunci model, recipe, task, K, parser, generation dan kriteria effect praktis sebelum membuka test.

Pemilihan checkpoint memakai mean macro-F1 dari task terpilih; invalid parse dan respons yang hilang akibat runtime dihitung salah. Tie-break melalui coverage lalu checkpoint lebih awal merupakan R/H yang harus dikunci. Teacher-forced loss tetap diagnostik. Laporkan F1/accuracy per task, parse coverage, unknown/abstention/exclusions, hasil berkelompok menurut subjek dan biaya. Jangan menetapkan kenaikan universal sebagai syarat sukses tanpa dasar pengukuran.

Proof of concept satu seed bersifat exploratory. Uncertainty dihitung melalui pasangan dan cluster subjek/recording sesuai dokumen 05, bukan confidence interval dari frame seolah independen. Multi-seed diperlukan untuk klaim robustness terhadap inisialisasi, bukan syarat awal feasibility. Hasil negatif bermakna jika data, labels, training dan presisi pengukuran memadai; bug atau optimasi yang tidak berhasil menghasilkan kesimpulan inconclusive.

Checkpoint menyimpan projector state dan alpha, condition/K/seed, epoch/update/selection metrics serta hash upstream, tokenizer, cache, QA, split, task, recipe, prompt, parser, model dan runtime. Last state menambahkan optimizer, scheduler, scaler, RNG, sampler, urutan data dan skipped updates. Resume pada epoch boundary menjadi default; resume mid-accumulation membutuhkan gradient dan batch-offset state yang lengkap. Gunakan safe weights-only loading, strict state_dict/lineage dan atomic save pada artefak run baru, tanpa unsafe unpickle fallback.

Log tahap L mengikuti [dokumen 05 §12.2–12.3](05_evaluasi_testing.md#122-kontrak-log-minimum-untuk-analisis-pascarun): answer-only CE, projector gradients/LR/skipped updates, generated validation scores, raw IDs/text, parse/confusion/failures serta donor coverage dengan reference IDs. Bandingkan dengan hasil probe U untuk diagnosis representasi versus alignment. Perbaikan projector/prefix dilakukan train/validation pada matched protocol; hasil test tidak dipakai menuning ulang. Log lengkap tidak menjamin frozen LLM dapat memakai semua informasi yang terbaca probe.

## 10. Kontrol grounding dan keluaran mentah

### 10.1 Pasangan dengan reference answer yang benar

Kontrol grounding P0 memakai pasangan window dari **recording berbeda, keluarga aktivitas sama, dan reference answer berbeda**. Pertanyaan, task, body part dan reference frame tetap sama; interval waktu relatif harus kompatibel. C_text dan majority tetap dilaporkan. Donor map, cohort dan aturan compatibility dikunci sebelum test. Donor U diekstrak oleh tokenizer kondisi bersangkutan, kemudian projector dan generation dijalankan ulang. Setiap respons dinilai terhadap reference window asalnya, termasuk donor; pertanyaan tidak diubah menjadi petunjuk jawaban.

Laporkan **both-correct**: kedua jawaban pasangan benar; dan **correct-change**: kedua jawaban benar serta perubahan jawaban sesuai dua reference berbeda. Sertakan denominator pasangan eligible, jumlah recording/subjek unik, task/class coverage dan proporsi window yang mempunyai donor sah. Pada single-field task dengan reference berbeda, kedua metrik dapat berimpit; jangan memperlakukannya sebagai dua bukti independen. Jawaban berubah saja tidak membuktikan grounding. Donor dengan reference sama hanya invariant control, bukan bukti correct-change. Jika pasangan sah tidak tersedia, laporkan keterbatasan coverage; jangan menggantinya diam-diam dengan pengacakan latent.

Time/bin shuffle atau token drop hanya diagnostik out-of-distribution. Tidak semua task berubah label, sehingga penurunan skor bukan bukti kausalitas fisik. Intervensi arbitrary latent atau perubahan tanda velocity bukan physical counterfactual. Synthetic/reversed trajectory memerlukan audit sensor, target dan support terpisah; bukan kewajiban P0.

Explicit predicted numerical facts dan oracle GT facts adalah diagnostik terpisah. Jangan memasukkan answer category sebagai “facts”. Oracle bukan kondisi deployment atau masukan utama U.

### 10.2 Inferensi dan parser

~~~text
infer(bundle,RPC_window,question) -> raw_ids,raw_text,parse_status,parsed_answer
attach_predicted_evidence(record,physical_prediction,recipe) -> evidence_metadata
~~~

Inferensi memakai context sensor nyata t−3..t, tanpa duplikasi missing frame. Pertanyaan di luar registry mendapat status unsupported, berbeda dari jawaban salah pada QA eligible. Evidence dari U readout atau H decoder diberi **evidence_source** yang jelas; angka dari H tidak diklaim berasal U. Evidence tidak dimasukkan ke main prompt dan tidak digunakan untuk memperbaiki raw answer atau skor.

Parser stdlib JSON menolak duplicate keys, task mismatch, enum invalid, missing/extra keys, banyak objek, fences dan prose; whitespace luar diperbolehkan. Registry/parser sama pada validation dan test. Tidak ada perbaikan yang menaikkan skor. Jika deployment memakai constrained decoding atau rule correction, varian raw/corrected dilaporkan terpisah.

Output record menyimpan condition, K, window, support, lineage, raw IDs/text, hasil parse, status/error dan sensor-predicted evidence jika tersedia. GT answer/reference hanya diakses evaluator terpisah. Respons invalid pada QA eligible dihitung salah; denominator tidak diperkecil. Kontradiksi evidence boleh ditandai tetapi tidak mengganti jawaban.

## 11. Sumber daya RTX 3060 dan gerbang biaya

**T — Bukan hasil pengukuran VRAM:** config dan susunan Qwen referensi dengan tied embeddings menghasilkan sekitar 1,544 miliar unique parameters. Pada dua byte per parameter, bobot sekitar 3,087 GB desimal atau 2,875 GiB. [Config resmi](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) dan [model source](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py). Jumlah parameter, tying, dtype dan memori model aktual tetap diperiksa.

| Komponen ilustratif, B=1/S=256 | Ukuran |
|---|---:|
| Input embeddings: 256×1536×2 byte | 0,75 MiB |
| Satu FFN intermediate per layer: 256×8960×2 byte | 4,375 MiB |
| Full logits fp32: 256×151936×4 byte | 148,375 MiB |
| Projector fp32: parameters, gradients dan dua Adam moments | 14,039 MiB |
| KV cache: 28 layers, dua KV heads, head width 128, dua byte | 7 MiB |

FFN mempunyai beberapa intermediate, sementara copies, workspaces, CUDA context, allocator dan saved tensors untuk backward menambah peak. KV cache dimatikan saat training, tetapi aktivasi untuk backward terhadap input tetap diperlukan. **Tidak adanya optimizer LLM bukan berarti backward LLM tidak memakai memori.** Tabel bukan jumlah total peak atau jaminan muat pada 12 GB.

Tahap L hanya memerlukan cache U: pada K=8/16/32, masing-masing 4/8/16 KiB per window fp16; H/M tidak harus resident. Tahap C memerlukan fixed H sebesar 184 KiB per window dan M yang dapat dibentuk saat dibaca. H bukan debug-only cache. Gunakan bounded shards dan lazy loading untuk RAM 16 GB; dataset 50 GB tidak dimuat penuh. Cache U tidak digandakan per paraphrase, dan bobot Qwen tidak disalin per kondisi.

Profil dilakukan pada perangkat eksperimen setelah warm-up dan CUDA synchronization. Catat loaded memory, peak allocated/reserved saat backward/update/generation, CPU RSS, QA per detik, panjang prefix/jawaban, rejection rate, median/p95 step time, validation time dan biaya I/O. Estimasi durasi training berasal dari pilot, bukan ukuran disk dataset. Jika tidak muat atau tidak stabil, audit pemuatan ganda, logits, cache dan precision, lalu uji native checkpointing. Perubahan capacity, panjang sequence atau protocol diterapkan sepadan sebelum test. Jangan menjanjikan durasi atau mengganti LLM hanya pada satu kondisi.

## 12. Gerbang integrasi dan artefak

Nama modul, model, config dan API di atas adalah rencana; belum dibuat atau dilatih oleh revisi dokumentasi. Konsep loop, projector dan QA MM-Fi boleh digunakan ulang setelah audit kompatibilitas, bukan dimensi/checkpoint historisnya. Implementasi kelak memakai .venv/uv dan Pure-PyTorch; sanity check wajib lulus sebelum training panjang.

| Gate | Bukti yang dibutuhkan |
|---|---|
| Perbandingan ilmiah | C_base/C_kin memakai full-H fidelity, M-versus-U auxiliary source, input/label/capacity/exposure yang dikontrol |
| Input dan provenance | U aktual tanpa GT; perubahan target tidak mengubah prefix; mask dan waktu bin sama |
| Freeze dan autograd | Hanya projector diperbarui; gradient terhadap input LLM ada; upstream checksums tetap |
| Reliabilitas task | Tugas temporal dan relasional sah; klaim event order memerlukan onset gate independen |
| QA dan loss | Satu registry; answer-only labels; causal shift sekali; accumulation per QA |
| Generation | Prefix, raw first logits, decode dan padding ekuivalen; parser fixtures lulus |
| Fairness alignment | Projector independen dengan K, initial state, prompts, samples, updates dan generation sepadan |
| Grounding | Pasangan same-activity dari recording berbeda dengan reference berbeda; both-correct/correct-change dan coverage |
| Representasi | Matched postfreeze M/U probes; Z sekunder; sumber readout tidak tertukar |
| Biaya dan test | Precision/backward/RAM diprofil; budget, panel dan kriteria dikunci sebelum test |

Artefak final disimpan per condition/K/seed: config/lineage, initial/best/last projector, training history, raw validation/test records, metrics, probes, coverage dan resource profile. Inference bundle menunjuk frozen encoder/motion/tokenizer, physical head opsional, projector, Qwen revision, calibration, joint map, normalizers, tasks, recipes, bin policy, prompt, parser dan generation hashes. Jangan menduplikasi bobot Qwen atau menimpa laporan MM-Fi.

Jika pelestarian informasi U membaik tetapi QA tidak, laporkan batas alignment/pemanfaatan. Jika QA membaik tanpa bukti retention, mekanisme pelestarian belum didukung. Jika baseline sudah cukup baik, hipotesis bottleneck tidak terbukti pada scope itu. Kesimpulan mengikuti dokumen 05 dan data, bukan tujuan membuat C_kin selalu menang.
