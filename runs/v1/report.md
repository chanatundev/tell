# Validation report (main run)

- val AUROC: **1.000**
- recall at threshold 0.0214: 157/157 = 1.000 (95% CI 0.976-1.000)
- false-alarm rate on val reals: 13/278 = 0.047 (95% CI 0.028-0.078)
- temperature 1.572, val NLL 0.014

| source | kind | flagged / n | rate (Wilson 95%) |
|---|---|---|---|
| gpt4o_receipt | recall | 157/157 | 1.000 (0.976-1.000) |
| cord_v2 | false-alarm | 5/200 | 0.025 (0.011-0.057) |
| commons | false-alarm | 8/78 | 0.103 (0.053-0.190) |

Caveat: epoch, temperature and threshold were chosen on this same val set, so these numbers are optimistic.
This project is a prototype and is not financial advice.