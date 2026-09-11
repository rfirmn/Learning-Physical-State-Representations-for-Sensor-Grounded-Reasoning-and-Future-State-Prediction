15. Future-state prediction

Task utama:

Given observations from (t-k) to (t), predict the physical state at (t+\Delta).

Contoh:

Input:
Radar(t-10 ... t)
Output:
Position(t+1)
Velocity(t+1)
Trajectory(t+1 ... t+k)

Untuk human motion:

Radar sequence
      ↓
Encoder
      ↓
Physical state
      ↓
Dynamics model
      ↓
Future skeleton / trajectory

Ini menjadi komponen predictive modeling dalam penelitian.
