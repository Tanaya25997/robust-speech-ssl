# robust-speech-ssl
Self-supervised speech representation learning on noisy/degraded audio

Stage 1 — sanity check
  train.clean.100 (100 hours)
  mild augmentation
  just check loss goes down, representations make sense

Stage 2 — research experiments  
  train.clean.100 + train.other.500 (600 hours)
  full augmentation pipeline
  proper evaluation on downstream tasks

Stage 3 — full scale
  all 960 hours + augmentation
  final numbers for the paper


Stage 1 (100 hours data, ~4-8hrs training)
  → $62.40 × 6hrs average = ~$375

Stage 2 (600 hours data, ~12-24hrs training)
  → $62.40 × 18hrs average = ~$1,125

Stage 3 (960 hours data, ~20-40hrs training)
  → $62.40 × 30hrs average = ~$1,870

Total across all stages    →  ~$3,370