# LLM-judge reliability

30 answers; judge `local:Qwen/Qwen2.5-1.5B-Instruct` says *faithful* when score ≥ 0.5. Items a rater marked *unsure* are excluded from that rater's comparisons.

| Comparison | n | Agreement | Cohen's κ [95% CI] | Rater yes-rate | Judge yes-rate |
|---|---|---|---|---|---|
| human vs judge (app_truncated) | 26 | 0.65 | 0.00 [0.00–0.00] | 0.65 | 1.00 |
| human vs judge (full_context) | 26 | 0.65 | 0.00 [0.00–0.00] | 0.65 | 1.00 |
| claude vs judge (app_truncated) | 30 | 0.60 | 0.00 [0.00–0.00] | 0.60 | 1.00 |
| claude vs judge (full_context) | 30 | 0.60 | 0.00 [0.00–0.00] | 0.60 | 1.00 |
| human vs claude (inter-annotator) | 26 | 0.69 | 0.35 [-0.04–0.69] | – | – |

## human vs judge (app_truncated)

Excluded as unsure: j01, j09, j12, j16

| | judge yes | judge no |
|---|---|---|
| **human yes** | 17 | 0 |
| **human no** | 9 | 0 |

Threshold sweep: 0.3: κ=0.00, 0.4: κ=0.00, 0.5: κ=0.00, 0.6: κ=0.00, 0.7: κ=0.00, 0.8: κ=-0.13, 0.9: κ=-0.13

Disagreements: j02 (human=no, judge=0.9), j07 (human=no, judge=0.7), j11 (human=no, judge=0.9), j13 (human=no, judge=0.9), j19 (human=no, judge=0.7), j26 (human=no, judge=0.9), j27 (human=no, judge=0.9), j28 (human=no, judge=0.9), j29 (human=no, judge=0.9)

## human vs judge (full_context)

Excluded as unsure: j01, j09, j12, j16

| | judge yes | judge no |
|---|---|---|
| **human yes** | 17 | 0 |
| **human no** | 9 | 0 |

Threshold sweep: 0.3: κ=0.00, 0.4: κ=0.00, 0.5: κ=0.00, 0.6: κ=0.00, 0.7: κ=0.00, 0.8: κ=-0.13, 0.9: κ=-0.13

Disagreements: j02 (human=no, judge=0.9), j07 (human=no, judge=0.7), j11 (human=no, judge=0.9), j13 (human=no, judge=0.9), j19 (human=no, judge=0.7), j26 (human=no, judge=0.9), j27 (human=no, judge=0.9), j28 (human=no, judge=0.9), j29 (human=no, judge=0.9)

## claude vs judge (app_truncated)

Excluded as unsure: none

| | judge yes | judge no |
|---|---|---|
| **claude yes** | 18 | 0 |
| **claude no** | 12 | 0 |

Threshold sweep: 0.3: κ=0.00, 0.4: κ=0.00, 0.5: κ=0.00, 0.6: κ=0.00, 0.7: κ=0.00, 0.8: κ=0.12, 0.9: κ=0.06

Disagreements: j01 (claude=no, judge=0.9), j04 (claude=no, judge=0.7), j08 (claude=no, judge=0.9), j10 (claude=no, judge=0.9), j11 (claude=no, judge=0.9), j13 (claude=no, judge=0.9), j15 (claude=no, judge=0.7), j17 (claude=no, judge=0.7), j19 (claude=no, judge=0.7), j26 (claude=no, judge=0.9), j27 (claude=no, judge=0.9), j29 (claude=no, judge=0.9)

## claude vs judge (full_context)

Excluded as unsure: none

| | judge yes | judge no |
|---|---|---|
| **claude yes** | 18 | 0 |
| **claude no** | 12 | 0 |

Threshold sweep: 0.3: κ=0.00, 0.4: κ=0.00, 0.5: κ=0.00, 0.6: κ=0.00, 0.7: κ=0.00, 0.8: κ=0.12, 0.9: κ=0.12

Disagreements: j01 (claude=no, judge=0.9), j04 (claude=no, judge=0.7), j08 (claude=no, judge=0.9), j10 (claude=no, judge=0.9), j11 (claude=no, judge=0.9), j13 (claude=no, judge=0.9), j15 (claude=no, judge=0.75), j17 (claude=no, judge=0.7), j19 (claude=no, judge=0.7), j26 (claude=no, judge=0.9), j27 (claude=no, judge=0.9), j29 (claude=no, judge=0.9)

## human vs claude

Disagreements: j02 (human=no, claude=yes), j04 (human=yes, claude=no), j07 (human=no, claude=yes), j08 (human=yes, claude=no), j10 (human=yes, claude=no), j15 (human=yes, claude=no), j17 (human=yes, claude=no), j28 (human=no, claude=yes)

κ guide (Landis & Koch): <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, >0.80 almost perfect. κ is 0 by construction when a rater gives the same label to every item. With n≤30 the CI is wide.
