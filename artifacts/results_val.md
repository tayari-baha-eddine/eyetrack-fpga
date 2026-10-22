Partition : val (60 images). Erreurs en pixels de la ROI 400x400.

| Modele | moyenne | mediane | p95 | <1 % | <2 % | <5 % |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (seuil 195, echecs 0.0%) | 64.83 | 56.54 | 109.88 | 0.0% | 0.0% | 1.7% |
| MLP flottant | 4.06 | 3.54 | 9.35 | 56.7% | 90.0% | 100.0% |
| MLP entier INT8/INT32 | 5.17 | 4.66 | 9.97 | 41.7% | 75.0% | 100.0% |

Ecart flottant / entier : moyenne 3.222 px, maximum 5.879 px
