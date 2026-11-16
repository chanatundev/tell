# Held-out twin evaluation (2026-11-15T14:55:53)

50 test pairs (real Commons photo + AI twin from a newer generator). Model 59f86ee31879, git 341e0b27a7655855b9f57c9180dc50468e74f83f, frozen 2026-11-15T19:53:46.037669+00:00. frozen.json committed in git: True.

| method | paired accuracy (bootstrap 95% CI) | recall on twins (Wilson) | false alarms on reals (Wilson) | AUROC (all images) |
|---|---|---|---|---|
| **Tell** (frozen threshold) | 0.780 (0.66-0.88) | 37/50 = 0.74 (0.60-0.84) | 15/50 = 0.30 (0.19-0.44) | 0.740 |
| freq_lr | 0.660 (0.52-0.78) | 22/50 = 0.44 (0.31-0.58) | 14/50 = 0.28 (0.17-0.42) | 0.634 |
| checklist_lenient | 0.550 (0.49-0.61) | 11/50 = 0.22 (0.13-0.35) | 8/50 = 0.16 (0.08-0.29) | 0.533 |
| checklist_strict | 0.490 (0.42-0.56) | 40/50 = 0.80 (0.67-0.89) | 41/50 = 0.82 (0.69-0.90) | 0.483 |

In-distribution (GPT-4o val split): AUROC 1.000, recall 1.000, false-alarm 0.047 (val was also used to pick epoch/threshold, so optimistic).

Paired accuracy = share of pairs where the twin scores strictly higher than its real partner (ties = 0.5); chance = 0.5. Baseline thresholds: freq LR flagged at its own 5%-val-false-alarm threshold; checklist flags at >= 1 rule.

### False alarms on the 50 reals by country

| group | false alarms / n | Wilson 95% |
|---|---|---|
| Japan | 2/5 | 0.12-0.77 |
| United States | 1/5 | 0.04-0.62 |
| Poland | 2/4 | 0.15-0.85 |
| China | 0/3 | 0.00-0.56 |
| Germany | 0/3 | 0.00-0.56 |
| Netherlands | 0/3 | 0.00-0.56 |
| Taiwan | 0/3 | 0.00-0.56 |
| United Kingdom | 0/2 | 0.00-0.66 |
| Australia | 1/1 | 0.21-1.00 |
| Austria | 0/1 | 0.00-0.79 |
| Brazil | 0/1 | 0.00-0.79 |
| Curaçao | 0/1 | 0.00-0.79 |
| Czech Republic | 1/1 | 0.21-1.00 |
| France | 1/1 | 0.21-1.00 |
| Hong Kong | 0/1 | 0.00-0.79 |
| India | 0/1 | 0.00-0.79 |
| Indonesia | 1/1 | 0.21-1.00 |
| Ireland | 0/1 | 0.00-0.79 |
| Israel | 0/1 | 0.00-0.79 |
| Italy | 1/1 | 0.21-1.00 |
| Latvia | 1/1 | 0.21-1.00 |
| Nigeria | 1/1 | 0.21-1.00 |
| Norway | 0/1 | 0.00-0.79 |
| Philippines | 0/1 | 0.00-0.79 |
| Russia | 1/1 | 0.21-1.00 |
| South Korea | 0/1 | 0.00-0.79 |
| Spain | 1/1 | 0.21-1.00 |
| Thailand | 0/1 | 0.00-0.79 |
| Ukraine | 1/1 | 0.21-1.00 |
| Vietnam | 0/1 | 0.00-0.79 |

### False alarms on the 50 reals by capture

| group | false alarms / n | Wilson 95% |
|---|---|---|
| phone photo | 9/33 | 0.15-0.44 |
| flat scan | 6/17 | 0.17-0.59 |

### False alarms on the 50 reals by script

| group | false alarms / n | Wilson 95% |
|---|---|---|
| English (Latin) | 3/12 | 0.09-0.53 |
| Japanese (kanji/kana) | 2/5 | 0.12-0.77 |
| German (Latin) | 0/4 | 0.00-0.49 |
| Polish (Latin) | 2/4 | 0.15-0.85 |
| Dutch (Latin) | 0/3 | 0.00-0.56 |
| Simplified Chinese (hans) | 0/3 | 0.00-0.56 |
| Traditional Chinese (hant) | 0/3 | 0.00-0.56 |
| Czech (Latin) | 1/1 | 0.21-1.00 |
| English (Latin) with Vietnamese currency | 0/1 | 0.00-0.79 |
| English with Papiamento (Latin) | 0/1 | 0.00-0.79 |
| French (Latin) | 1/1 | 0.21-1.00 |
| Hebrew (right-to-left) | 0/1 | 0.00-0.79 |
| Indonesian (Latin) | 1/1 | 0.21-1.00 |
| Italian (Latin) | 1/1 | 0.21-1.00 |
| Korean (Hangul) | 0/1 | 0.00-0.79 |
| Latvian (Latin) | 1/1 | 0.21-1.00 |
| Norwegian (Latin) | 0/1 | 0.00-0.79 |
| Portuguese (Latin) | 0/1 | 0.00-0.79 |
| Russian (Cyrillic) | 1/1 | 0.21-1.00 |
| Spanish (Latin) | 1/1 | 0.21-1.00 |
| Thai (Thai script) | 0/1 | 0.00-0.79 |
| Traditional Chinese (hant) with English | 0/1 | 0.00-0.79 |
| Ukrainian (Cyrillic) | 1/1 | 0.21-1.00 |

### Most confident failures (files in results/examples/; not displayed here)

| kind | id | score | country | capture | file |
|---|---|---|---|---|---|
| miss (AI twin scored low) | r0753 | 0.000 | United Kingdom | phone photo | examples/miss_r0753_twin.jpg |
| miss (AI twin scored low) | r0388 | 0.000 | Netherlands | phone photo | examples/miss_r0388_twin.jpg |
| miss (AI twin scored low) | r0053 | 0.000 | United States | flat scan | examples/miss_r0053_twin.jpg |
| false alarm (real scored high) | r0093 | 1.000 | Indonesia | phone photo | examples/falsealarm_r0093_real.jpg |
| false alarm (real scored high) | r0691 | 0.985 | United States | phone photo | examples/falsealarm_r0691_real.jpg |
| false alarm (real scored high) | r0326 | 0.955 | Japan | phone photo | examples/falsealarm_r0326_real.jpg |

**Caveats:** small, team-made set; synthetic twins are not real fraud; a flag is not proof of fraud; group rows are warning signs, not proof of unfairness. This project is a prototype and is not financial advice.