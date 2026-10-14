# DPO signal probe — same-prompt / different-seed spread

Seeds: 42 43 44 45

## tc_clone_100

100 prompts · 100 with >=2 seeds (0 excluded, too few seeds) · **99/100 pairable** (99.0%)

| signal | prompts flagged |
|---|---|
| wer | 52 |
| spk_sim | 90 |
| runaway | 1 |
| nvpa | 69 |

## sn_clone_100

100 prompts · 100 with >=2 seeds (0 excluded, too few seeds) · **95/100 pairable** (95.0%)

| signal | prompts flagged |
|---|---|
| wer | 42 |
| spk_sim | 86 |
| runaway | 1 |
| nvpa | 31 |

## br_clone_100

100 prompts · 100 with >=2 seeds (0 excluded, too few seeds) · **99/100 pairable** (99.0%)

| signal | prompts flagged |
|---|---|
| wer | 46 |
| spk_sim | 95 |
| runaway | 0 |
| nvpa | 59 |

## la_clone_100

100 prompts · 100 with >=2 seeds (0 excluded, too few seeds) · **98/100 pairable** (98.0%)

| signal | prompts flagged |
|---|---|
| wer | 52 |
| spk_sim | 95 |
| runaway | 0 |
| nvpa | 60 |

## vi_probe_100

100 prompts · 100 with >=2 seeds (0 excluded, too few seeds) · **95/100 pairable** (95.0%)

| signal | prompts flagged |
|---|---|
| wer | 50 |
| spk_sim | 88 |
| runaway | 0 |

# Tag discrimination (blind judge, what was heard at the tag position)

| expected tag | n | hit rate | laughter | breath | gasp | sigh | cough | throat_clear | sniff | other | none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| throatclearing (tc_clone_100) | 400 | 71.5% | 18 | 2 | 1 | 6 | 70 | 286 | 2 | 9 | 6 |
| sniff (sn_clone_100) | 400 | 12.5% | 22 | 126 | 2 | 15 | 8 | 4 | 50 | 5 | 168 |
| breathing (br_clone_100) | 400 | 26.2% | 17 | 105 | 51 | 72 | 7 | 0 | 4 | 7 | 137 |
| laughter (la_clone_100) | 400 | 63.5% | 254 | 19 | 0 | 16 | 3 | 7 | 4 | 10 | 87 |

| expected tag | prompts all-correct | **mixed (DPO pairs)** | all-wrong (needs data) | <2 seeds judged | wrong labels in mixed | wrong labels in all-wrong | hit rate by position | naturalness hit / miss |
|---|---|---|---|---|---|---|---|---|
| throatclearing | 30 | **69** | 1 | 0 | cough 69, laughter 16, other 9, none 6 | laughter 2, sigh 1, cough 1 | beginning 75% (n=136), end 67% (n=132), middle 72% (n=132) | 3.85 / 3.37 |
| sniff | 2 | **31** | 67 | 0 | none 29, breath 26, laughter 11, sigh 8 | none 139, breath 100, laughter 11, sigh 7 | beginning 5% (n=136), end 23% (n=132), middle 10% (n=132) | 3.72 / 2.43 |
| breathing | 1 | **59** | 40 | 0 | none 56, sigh 38, gasp 26, laughter 11 | none 81, sigh 34, gasp 25, cough 6 | beginning 15% (n=136), end 23% (n=132), middle 41% (n=132) | 3.69 / 2.32 |
| laughter | 33 | **60** | 7 | 0 | none 67, sigh 15, breath 15, other 9 | none 20, breath 4, cough 1, sigh 1 | beginning 40% (n=136), end 82% (n=132), middle 69% (n=132) | 3.83 / 2.04 |

