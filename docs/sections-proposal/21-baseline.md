21. Baseline

Baseline dibangun secara bertingkat.

Model	Sensor Encoder	Dynamics	LLM	Adaptation
Text baseline	—	—	✓	—
Structured sensor	—	—	✓	—
Sensor encoder	✓	—	—	—
Sensor encoder + dynamics	✓	✓	—	—
Sensor representation + LLM	✓	—	✓	Frozen
Sensor representation + LLM + PEFT	✓	—	✓	LoRA
Full proposed pipeline	✓	✓	✓	Optional

Dengan demikian dapat diketahui kontribusi:

Representation
vs
Dynamics
vs
LLM
vs
LLM Adaptation
