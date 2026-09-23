3. Latar belakang

Large Language Models telah menunjukkan kemampuan reasoning yang kuat pada informasi linguistik. Namun, kemampuan tersebut tidak secara otomatis berarti bahwa model memahami keadaan dan dinamika dunia fisik.

Sensor memberikan informasi yang berbeda dari bahasa.

Misalnya mmWave radar dapat memberikan:

x
y
z
velocity / Doppler
timestamp
SNR

Informasi tersebut menggambarkan physical observation secara langsung.

Namun, terdapat beberapa tahapan yang berbeda antara observation dan reasoning:

Sensor Observation
       ↓
Physical State
       ↓
State Transition / Dynamics
       ↓
Future State
       ↓
Reasoning

Model yang hanya mampu melakukan:

“What activity is being performed?”

belum tentu mampu melakukan:

“Bagaimana postur dan konfigurasi tubuh subjek saat ini?”

“Apakah kedua tangan subjek sedang terangkat atau merapat ke torso?”

“Bagaimana geometri artikulasi tubuh akan berubah dalam rentang waktu berikutnya?”

“Di mana posisi relatif sendi pergelangan tangan setelah 0.8 detik?”

“Bagaimana perubahan penalaran jika urutan dinamika fisik sensorik diintervensi?”

Pertanyaan tersebut membutuhkan pemahaman terhadap:

* spatial information;
* temporal information;
* velocity;
* trajectory;
* object relations;
* state transitions;
* physical dynamics.

Oleh karena itu, penelitian ini tidak hanya menanyakan apakah sensor dapat dimasukkan ke dalam LLM.

Pertanyaan yang lebih fundamental adalah:

Apakah learned sensor representations mampu mempertahankan informasi physical state dan dynamics yang diperlukan untuk reasoning dan future-state prediction?

Pertanyaan tersebut merupakan langkah menuju world modeling.
