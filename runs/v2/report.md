POST-UNSEAL DEVELOPMENT — NOT FRESH HELDOUT

# Validation report (main run)

- val AUROC: **1.000**
- recall at threshold 0.0024: 164/164 = 1.000 (95% CI 0.977-1.000)
- false-alarm rate on val reals: 37/750 = 0.049 (95% CI 0.036-0.067)
- temperature 0.820, val NLL 0.009

| source | kind | flagged / n | rate (Wilson 95%) |
|---|---|---|---|
| gpt4o_receipt | recall | 157/157 | 1.000 (0.976-1.000) |
| cord_v2 | false-alarm | 4/200 | 0.020 (0.008-0.050) |
| commons | false-alarm | 4/78 | 0.051 (0.020-0.125) |
| wildreceipt | false-alarm | 29/472 | 0.061 (0.043-0.087) |
| newgen_fakes | recall | 7/7 | 1.000 (0.646-1.000) |

Caveat: epoch, temperature and threshold were chosen on this same val set, so these numbers are optimistic.
This project is a prototype and is not financial advice.