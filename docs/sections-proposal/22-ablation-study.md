22. Ablation study

Studi ablasi dirancang untuk mengisolasi dan memahami kontribusi masing-masing komponen fisik, temporal, dan arsitektural:

```text
Full Pipeline (B4)
    │
    ├── 1. Ablasi Kanal Sensor (Input Modality)
    │     ├── Tanpa Doppler (Menghapus kecepatan radial v_d)
    │     ├── Tanpa SNR (Menghapus intensitas pantulan radar)
    │     └── Spatial-Only (Hanya koordinat 3D x, y, z)
    │
    ├── 2. Ablasi Pemodelan Dinamika Masa Depan
    │     ├── B4 (Predicted Dynamics) vs B3P (Persistence Baseline)
    │     └── B4 vs B3 (Historical Only, tanpa token masa depan)
    │
    ├── 3. Ablasi Konteks Temporal & Horizon
    │     ├── Variasi Konteks Historis T_in ∈ {8, 16} (0.8s vs 1.6s)
    │     └── Variasi Horizon Prediksi T_out ∈ {4, 8} (0.4s vs 0.8s)
    │
    └── 4. Ablasi Arsitektur Proyektor Penyelaras
          ├── Linear Projection tunggal (W * z)
          └── Two-Layer MLP Projector dengan aktivasi non-linear GELU
```

### Tujuan Ilmiah:
Tujuannya bukan sekadar mengejar angka akurasi tertinggi, melainkan mengungkap:
1. **Peran Sinyal Fisik Radar:** Sejauh mana kecepatan Doppler berkontribusi dalam membedakan pergerakan artikulasi cepat dibandingkan fitur posisi spasial murni?
2. **Kebutuhan Model Dinamika:** Apakah proyeksi masa depan eksplisit dari Dynamics Model memberikan keuntungan penalaran yang signifikan di atas model *persistence* statis?
3. **Kapasitas Pemetaan Antarmuka:** Apakah adapter proyektor non-linear (Two-Layer MLP) cukup menjembatani diskrepansi representasi tanpa perlu mengubah bobot model bahasa (LLM)?
