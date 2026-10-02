# Rekomendasi Model Encoder dan Dynamics

**Judul kerja:** Sensor-Grounded Kinematic Dynamics Representation for Temporal Reasoning  
**Status:** hasil telaah literatur dan keputusan desain awal  
**Tanggal:** 2 Oktober 2026  
**Hardware target:** RTX 3060 12 GB, RAM 16 GB

## 1. Keputusan utama

Pipeline yang paling layak untuk diteruskan:

    Radar point cloud RPC
          |
          v
    Lightweight P4Transformer-derived encoder
          |
          v
    Explicit physical state
    (joints, root trajectory, feature, confidence)
          |
          v
    Dual-stream spatial-temporal dynamics encoder
          |
          v
    Kinematic probes and time-aligned representation
          |
          v
    Frozen LLM alignment and temporal reasoning

Keputusan ini berarti:

1. Encoder utama menggunakan keluarga P4Transformer pada radar point cloud RPC karena ada bukti langsung di M4Human.
2. Model dinamika utama bukan decoder yang hanya memprediksi latent masa depan, tetapi encoder dinamika yang mempertahankan output pada setiap timestep.
3. Forecasting dipertahankan sebagai baseline atau auxiliary diagnostic, bukan sebagai definisi keberhasilan riset.

## 2. Temuan yang telah diverifikasi

### M4Human sesuai untuk tujuan dinamika

M4Human menyediakan radar mmWave, motion capture, global trajectory, dan 50 aktivitas manusia. Aktivitasnya mencakup walking cepat/lambat, gerak melengkung, lunge, kick, badminton, ping-pong, volleyball, dan boxing. Dataset tersebut memiliki sekitar 661 ribu frame dan 20 subjek.

Radar dan RGB-D direkam sekitar 12 Hz, sedangkan Vicon sekitar 100 Hz. Frame Vicon terdekat dipasangkan ke frame sensor. Karena itu, velocity dan acceleration harus dihitung dari mocap yang telah dihaluskan, kemudian disejajarkan ke timestamp radar.

Pada 12 Hz, frekuensi Nyquist sekitar 6 Hz. Penelitian tidak boleh mengklaim pengukuran presisi untuk seluruh akselerasi mikro. Klaim yang aman adalah perubahan kinematik pada skala gerak manusia: perpindahan, arah, percepatan/perlambatan, fase, trajectory, dan transisi gerak.

Sumber: [M4Human paper](https://arxiv.org/html/2512.12378v3), [M4Human repository](https://github.com/FanJunqiao/M4Human).

### RPC lebih realistis daripada RT untuk jalur utama

M4Human melaporkan P4Transformer RPC sekitar 7,17 ms dan 11,76 GFLOPs, sedangkan RT-Mesh sekitar 2,74 ms dan 2,60 GFLOPs. MVE P4Transformer untuk seluruh protokol dilaporkan sekitar 90,4 mm pada split acak, 140,8 mm pada cross-subject, dan 147,8 mm pada cross-action.

P4Transformer merupakan titik awal yang tervalidasi untuk RPC. RT-Mesh tetap penting sebagai ablation, tetapi representasi volume RT lebih besar dan lebih mahal untuk di-cache pada ratusan ribu frame. Dengan RTX 3060, RPC adalah kompromi utama yang lebih aman.

Error pose 7–15 cm juga menunjukkan bahwa velocity dan acceleration tidak boleh dihitung dari finite difference pose radar mentah. Model dinamika harus diberi target derivative dari mocap yang telah diproses dan, bila perlu, noise residual encoder saat training.

### Batas baseline M4Human

Baseline M4Human membuktikan bahwa konteks temporal pendek membantu estimasi pose/mesh. Itu belum sama dengan model dinamika kinematik yang secara eksplisit mempelajari velocity, acceleration, fase, trajectory, atau perubahan antar-joint. Ruang kontribusi riset berada pada tahap tersebut.

## 3. Model encoder sensor

### Pilihan utama: P4Transformer-derived RPC encoder

P4Transformer memodelkan video point cloud 4D tanpa point tracking. Konsepnya adalah mengelompokkan titik pada dimensi ruang dan waktu, mengurangi jumlah token, lalu memakai self-attention untuk menggabungkan informasi spasial dan gerak.

Sumber: [Point 4D Transformer, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf).

Yang dipakai adalah keluarga arsitekturnya, bukan klaim bahwa konfigurasi resmi M4Human harus disalin persis. Kita memerlukan varian ringan yang dapat dilatih dan diaudit pada RTX 3060.

### Input encoder

    RPC sequence: [T_enc, N, C]
    T_enc = 4 frame
    N     = 512–1.000 titik per frame
    C     = kanal sensor yang benar-benar tersedia

Keputusan praktis:

- gunakan RPC pada eksperimen utama;
- mulai dari empat frame agar kompatibel dengan baseline M4Human;
- gunakan delapan frame hanya sebagai ablation;
- gunakan 1.000 titik untuk kompatibilitas, atau 512 titik sebagai mode hemat memori;
- jangan mengasumsikan Doppler tersedia sebelum schema file RPC diaudit.

Jika radial velocity memang tersedia, lakukan ablation tanpa Doppler versus dengan Doppler. Jika tidak tersedia, hapus klaim tersebut dari desain eksperimen.

### Ukuran model awal

| Komponen | Nilai awal | Catatan |
|---|---:|---|
| Temporal context | 4 frame | sesuai baseline M4Human |
| Points/frame | 512–1.000 | 512 sebagai fallback memori |
| Hidden dimension | 256 | cukup untuk eksperimen awal |
| Transformer blocks | 4 | ringan untuk RTX 3060 |
| Attention heads | 4 atau 8 | harus membagi dimensi 256 |
| Precision | AMP fp16 | gunakan bila stabil |
| Feature output | 256-d | masuk ke dynamics model |

Empat blok dengan dimensi 256 adalah varian ringan yang diusulkan, bukan konfigurasi resmi yang boleh dikutip tanpa verifikasi kode.

### Output encoder

Encoder tidak boleh hanya mengeluarkan latent vector. Output minimal:

    state_t = {
        J_t : 22 x 3 joint coordinates,
        r_t : 3D global root translation,
        f_t : 256D sensor feature,
        c_t : confidence atau quality feature
    }

Gunakan format native M4Human bila annotation memang menggunakan 22 joint/SMPL-X. Jangan memaksa 17-joint MM-Fi ke dataset utama tanpa adapter dan dokumentasi mapping.

Simpan joint dalam bentuk root-relative untuk struktur tubuh dan root global untuk perpindahan seluruh tubuh. Output eksplisit memungkinkan kita membedakan error encoder dari error alignment LLM.

### Loss encoder

Mulai dari loss sederhana:

    L_encoder =
        joint position loss
      + root translation loss
      + bone-length consistency loss
      + weak temporal smoothness loss

Smoothness harus lemah. Regularisasi terlalu kuat dapat menghapus gerakan cepat hanya demi menghasilkan trajectory yang tampak halus. Jangan memasukkan acceleration loss besar pada tahap encoder.

### Model yang tidak dipilih sebagai utama

- PointNet++ atau set encoder: baseline murah, tetapi kurang kuat untuk struktur spasio-temporal.
- RT-Mesh: ablation efisien, bukan jalur utama karena format volume dan beban preprocessing.
- Point Transformer besar: belum tervalidasi langsung pada M4Human dan berisiko memboroskan memori.
- Point-MAE lama: dipertahankan sebagai baseline MM-Fi, bukan otomatis encoder final M4Human.

## 4. State untuk model dinamika

State yang diberikan ke dynamics model:

    x_t = [J_t_relative, r_t, f_t, c_t, d_t]

d_t hanya ada jika Doppler telah diverifikasi. State diproyeksikan ke dimensi 256 sebelum masuk ke dynamics model.

Velocity dan acceleration tidak perlu dihitung sebagai input utama dari pose prediksi yang noisy. Keduanya lebih aman dipelajari sebagai target dan auxiliary head. Finite difference dapat dipakai sebagai fitur tambahan dengan confidence, bukan sebagai ground truth.

## 5. Model dinamika

### Pilihan utama: dual-stream spatial-temporal dynamics encoder

Model utama adalah transformer kecil dengan dua jalur:

    state/joint tokens
          |
          +-- Spatial stream: relasi antar-joint pada setiap waktu
          |
          +-- Temporal stream: perubahan joint/state sepanjang waktu
                         |
                    controlled fusion
                         |
             time-aligned dynamics representation

Desain ini terinspirasi oleh DSTformer pada MotionBERT. Spatial attention memodelkan hubungan antar-joint dalam satu timestep, sedangkan temporal attention memodelkan pergerakan satu joint/state sepanjang waktu.

Sumber: [MotionBERT project](https://motionbert.github.io/), [MotionBERT ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV2023_paper.pdf).

### Konfigurasi awal

| Komponen | Nilai awal | Alasan |
|---|---:|---|
| Dynamics window | 32 frame | sekitar 2,7 detik pada 12 Hz |
| Extended ablation | 48 frame | sekitar 4 detik |
| Model width | 256 | sesuai encoder feature |
| Blocks | 4 | realistis pada RTX 3060 |
| Heads | 4 atau 8 | sesuai dimensi |
| Output | satu representasi per timestep | urutan tetap terjaga |

Mulai dari window 32. Window 48 hanya dipakai bila probe fase atau transisi belum stabil.

### Output dynamics model

    dynamics_t = {
        h_t     : representation per timestep,
        p_hat_t : reconstructed/denoised state,
        v_hat_t : velocity,
        a_hat_t : acceleration,
        phase_t : phase bila label valid,
        q_t     : quality/uncertainty bila diperlukan
    }

h_t harus time-aligned. Jangan melakukan global pooling sebelum evaluasi temporal. Jika seluruh window dipadatkan menjadi satu vector, shuffle-order dan reverse-order test tidak lagi bermakna.

Forecasting dapat ditambahkan sebagai head kecil atau auxiliary task, tetapi tidak boleh menggantikan h_t.

### Target dan loss dynamics

Target dibuat dari mocap 100 Hz:

1. audit timestamp dan alignment;
2. low-pass trajectory dengan cutoff di bawah batas Nyquist sensor;
3. hitung velocity dan acceleration pada domain mocap;
4. ambil/agregasikan target pada timestamp radar;
5. simpan filter dan metadata alignment.

Loss awal:

    L_dynamics =
        position loss
      + velocity loss
      + acceleration loss
      + kinematic consistency loss

Kinematic consistency menjaga hubungan diskrit antara posisi dan velocity. Loss phase hanya dipakai pada aktivitas yang memiliki definisi fase yang dapat dipertanggungjawabkan, misalnya siklus walking atau squat.

### JEPA atau masked temporal prediction

JEPA-style prediction layak sebagai auxiliary objective, bukan sebagai arsitektur ketiga yang wajib dibuat sejak awal. Urutannya:

1. latih supervised kinematic dynamics;
2. pastikan mengalahkan baseline;
3. tambahkan masked temporal objective sebagai ablation;
4. pertahankan hanya bila probe held-out meningkat.

### Mamba/SSM

Mamba valid sebagai kandidat sequence model linear-time, tetapi window awal hanya 32–48 frame. Manfaat efisiensi belum menjadi kebutuhan utama, sementara CUDA extension dapat menambah risiko pada lingkungan proyek.

Mamba baru layak diuji jika window diperpanjang jauh di atas 128 frame atau temporal transformer terbukti menjadi bottleneck. Mamba bukan model utama.

Sumber: [Mamba repository](https://github.com/state-spaces/mamba), [Mamba paper](https://arxiv.org/abs/2312.00752).

## 6. Baseline wajib

1. **Persistence:** state berikutnya sama dengan state terakhir.
2. **Constant velocity:** velocity terakhir diteruskan konstan.
3. **GRU atau TCN:** baseline murah untuk menguji apakah transformer benar-benar diperlukan.
4. **Temporal-only transformer:** tidak memiliki spatial attention antar-joint.
5. **Dual-stream spatial-temporal transformer:** model utama.

Ablation penting:

- state eksplisit versus state + sensor feature;
- tanpa Doppler versus dengan Doppler jika valid;
- pose-only versus pose + velocity + acceleration;
- pooled output versus time-aligned output;
- urutan asli versus shuffle versus reverse order;
- input bersih versus input dengan noise empiris encoder.

## 7. Dataset dan peran

| Dataset | Peran | Keputusan |
|---|---|---|
| M4Human | training/evaluasi utama radar-to-kinematics | wajib |
| AddBiomechanics | pretraining/probe kinematik dan force | opsional terarah |
| PhysioNet radar-force plate | validasi eksternal stabilitas/contact | validasi kecil |
| MM-Fi | sanity check dan baseline lama | bukan bukti utama |

### M4Human

Split harus dibuat berdasarkan recording/subjek/action sebelum sliding window. Minimal gunakan random split untuk debugging, cross-subject sebagai hasil utama, dan cross-action untuk menguji gerak baru. Laporkan protokol in-place, sit-in-place, dan non-in-place bila tersedia.

Repository resmi menyebut processed radar sekitar 50 GB dan konfigurasi benchmark multi-GPU. Karena itu, gunakan subset terstratifikasi dan cache fitur ringkas; jangan mencoba menjalankan seluruh benchmark resmi dengan konfigurasi multi-GPU di RTX 3060.

### AddBiomechanics

AddBiomechanics menyediakan posisi, velocity, acceleration, ground-contact force, joint torque, massa, inertia, dan center-of-mass kinematics pada lebih dari 24 juta frame dan 273 partisipan. Gunakan sebagai pretraining dynamics, physical probe, atau noise-robustness pretraining, bukan sebagai input radar langsung.

Pemetaan skeleton OpenSim ke joint M4Human/SMPL-X harus eksplisit dan divalidasi. Jika mapping tidak valid, jangan melaporkan force probe sebagai hasil langsung model radar.

Sumber: [AddBiomechanics dataset](https://www.addbiomechanics.org/download_data.html).

### PhysioNet

Dataset radar, motion capture, dan force plate one-legged stand cocok untuk external probe stabilitas dan foot-ground interaction. Karena aktivitasnya sempit, dataset ini bukan sumber training utama untuk walking, squat, atau boxing.

Sumber: [PhysioNet radar-force-plate data](https://physionet.org/content/?topic=force+plate).

## 8. Rencana eksperimen RTX 3060

### Fase 0 — audit data

Verifikasi jumlah titik, kanal RPC, Doppler, timestamp, frame hilang, mapping joint, unit koordinat, alignment radar-mocap, serta distribusi subjek dan gerakan. Arsitektur belum boleh dikunci sebelum audit ini selesai.

### Fase 1 — encoder

1. latih/adaptasi P4Transformer-derived encoder dengan AMP;
2. evaluasi MPJPE/MVE dan root trajectory error per protokol;
3. simpan checkpoint terbaik;
4. cache joint, root, feature, dan metadata.

Ukuran 661k x 256 x fp16 sekitar 338 MiB sebelum metadata dan output state. Caching fitur jauh lebih realistis daripada menyimpan ulang volume RT untuk seluruh frame.

### Fase 2 — dynamics

1. ukur persistence dan constant velocity;
2. latih GRU kecil;
3. latih temporal-only transformer;
4. latih dual-stream model;
5. evaluasi velocity, acceleration, phase, trajectory, dan order sensitivity;
6. lanjut ke LLM hanya jika model utama konsisten mengalahkan baseline pada held-out subject/action.

Batch dynamics kemungkinan berada di kisaran 32–64 karena input sudah berupa fitur ringkas. Nilai final tetap mengikuti pengukuran VRAM.

### Fase 3 — grounding evaluation

Probe yang dapat dihitung:

- arah root bergerak;
- joint yang paling banyak berpindah;
- mempercepat atau melambat;
- fase gerakan;
- transisi berdiri-squat-berdiri;
- perubahan trajectory;
- urutan kejadian.

LLM bukan satu-satunya evaluator. Jawaban harus dibandingkan dengan target kinematik atau parser terstruktur.

### Fase 4 — alignment LLM

Bandingkan:

1. LLM menerima state statis;
2. LLM menerima state dan dynamics representation;
3. LLM menerima dynamics representation yang di-shuffle;
4. LLM menerima urutan yang dibalik;
5. LLM menerima representasi dari model yang belum tervalidasi.

Klaim grounding hanya kuat jika kondisi asli mengungguli kondisi shuffle/reverse pada pertanyaan yang membutuhkan informasi temporal.

## 9. Kriteria keberhasilan

### Encoder

- mengalahkan baseline set encoder sederhana;
- error pose dan root trajectory dilaporkan pada cross-subject;
- output stabil pada gerakan dinamis;
- tidak ada leakage antar-window.

### Dynamics

- mengalahkan persistence dan constant velocity;
- meningkatkan probe velocity/acceleration atau phase;
- tetap lebih baik pada held-out subject/action;
- sensitif terhadap shuffle dan reverse order;
- tidak hanya mengandalkan label aktivitas.

### LLM grounding

- state + dynamics lebih baik daripada state-only;
- shuffle/reverse menurunkan performa secara terukur;
- jawaban konsisten dengan target kinematik;
- kemampuan tidak hilang total pada action atau subjek yang tidak dilihat.

Force, contact, dan torque hanya boleh menjadi klaim jika diuji pada dataset yang memang menyediakan label tersebut.

## 10. Risiko utama

- **Pose error menutupi dinamika:** gunakan derivative dari mocap yang dihaluskan dan noise residual encoder.
- **Leakage sliding window:** split sebelum membuat window dan simpan provenance.
- **Model hanya mengenali aktivitas:** gunakan same-action/different-phase, shuffle, reverse, dan state-only baseline.
- **Model terlalu besar:** gunakan RPC, caching, AMP, window 32, width 256, dan empat blok.
- **Klaim physical dynamics terlalu luas:** gunakan istilah kinematic dynamics kecuali force/contact/torque tersedia.
- **Doppler tidak tersedia:** audit schema lebih dulu dan jadikan posisi-only sebagai jalur reproducible.

## 11. Perubahan terhadap implementasi saat ini

Implementasi sekarang memakai Point-MAE/17-joint MM-Fi dan TemporalTransformerDynamics yang memetakan T_in latent ke T_out latent. Implementasi ini tidak perlu dihapus karena berguna sebagai baseline historis.

| Saat ini | Arah baru |
|---|---|
| MM-Fi sebagai dataset utama | M4Human sebagai dataset utama |
| 17 joint | format native M4Human atau adapter eksplisit |
| latent 384-d sebagai state | joint + root + feature + confidence |
| future latent forecasting | dynamics representation time-aligned |
| target future latent | position/velocity/acceleration/probe |
| LLM membaca forecast | LLM menalar perubahan state |

Urutan migrasi minimum:

1. tambahkan audit dan loader M4Human;
2. tambahkan adapter joint/root/feature;
3. buat dataset dynamics dengan target kinematik;
4. implementasikan model dual-stream di samping model lama;
5. evaluasi baseline;
6. baru ubah projector/LLM interface.

## 12. Keputusan final

### Encoder

Gunakan **P4Transformer-derived RPC encoder** dengan context empat frame, konfigurasi ringan, dan output eksplisit joint/root plus feature 256-d. Point-MAE lama menjadi baseline; RT-Mesh hanya ablation kecil.

### Dynamics

Gunakan **dual-stream spatial-temporal transformer** berukuran kecil, window 32 frame, width 256, empat blok, dan output time-aligned. GRU/TCN serta constant-velocity wajib menjadi baseline.

### Auxiliary

Masked temporal prediction atau JEPA-style objective baru ditambahkan setelah supervised kinematic objective bekerja. Jangan menjadikan JEPA, Mamba, RSSM, dan horizon forecasting sebagai komponen wajib sekaligus.

### Bahasa klaim

Gunakan istilah **sensor-grounded kinematic dynamics reasoning**. Gunakan **physical dynamics** hanya jika force/contact/torque benar-benar tersedia dan dievaluasi.

### Gate sebelum LLM

Jangan melatih projector LLM sebelum dynamics model mengalahkan persistence dan constant velocity, lolos cross-subject/cross-action probe, sensitif terhadap shuffle/reverse, dan meningkatkan reasoning temporal dibanding state-only baseline.

## 13. Referensi inti dan open source

1. [M4Human paper](https://arxiv.org/html/2512.12378v3) dan [official repository](https://github.com/FanJunqiao/M4Human)
2. [Point 4D Transformer, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf)
3. [MotionBERT project and code](https://motionbert.github.io/)
4. [MotionBERT, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV2023_paper.pdf)
5. [Mamba repository](https://github.com/state-spaces/mamba) dan [paper](https://arxiv.org/abs/2312.00752)
6. [AddBiomechanics dataset](https://www.addbiomechanics.org/download_data.html)
7. [RadarLLM, AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/37500/41462)
8. [SensorLLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.19/)
9. [LLaSA](https://arxiv.org/abs/2406.14498)
10. [PhysioNet radar-force-plate data](https://physionet.org/content/?topic=force+plate)

## Kesimpulan

Rancangan paling kuat bukan memilih model terbesar, melainkan memisahkan tiga pertanyaan: apakah radar dapat mengestimasi state tubuh, apakah state tersebut memuat dinamika kinematik, dan apakah frozen LLM benar-benar memakai dinamika itu untuk reasoning. Dengan hardware yang tersedia, kombinasi P4Transformer-derived RPC encoder dan dual-stream spatial-temporal dynamics encoder adalah pilihan paling masuk akal. Forecasting tetap dipakai sebagai baseline/diagnostic, sedangkan kontribusi utama berada pada representasi dinamika eksplisit, evaluasi temporal terkontrol, dan bukti bahwa bahasa ter-grounding pada perubahan keadaan fisik.
