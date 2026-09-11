11. Arsitektur yang diusulkan

Arsitektur utama:
mmWave Radar
                    │
                    ▼
           Sensor Preprocessing
                    │
                    ▼
              Sensor Encoder
                    │
                    ▼
       Physical State Representation (Z_t)
                    │
           ┌────────┴────────┐
           ▼                 ▼
    State Estimation    Dynamics Model
    (Pos, Vel, dll)   (Z_t ➔ Z_{t+1:t+k})
           │                 │
           └────────┬────────┘
                    ▼
         Future State Prediction 
        (Trajectory / Next States)
                    │
                    ▼
          [ Alignment Module / MLP ]  <-- (Lapisan Penyelaras ala SensorLLM)
                    │
                    ▼
              Frozen LLM
                    │
           ┌────────┴────────┐
           ▼                 ▼
       Reasoning         Explanation
   (Spatial/Relational) (Natural Language)


LLM tidak harus menjadi bagian dari physical dynamics model.

Ia berfungsi sebagai reasoning layer yang menggunakan physical representation.
