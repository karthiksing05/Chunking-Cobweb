
### Qwen3-8B syn: all (variant, thr) by dev margin (top 6)
- v5 thr=0.01 dev margin -0.140
- v1 thr=0.01 dev margin -0.147
- v2 thr=0.01 dev margin -0.153
- v5 thr=0.02 dev margin -0.160
- v1 thr=0.02 dev margin -0.160
- v4 thr=0.01 dev margin -0.167
chosen on dev: v5 thr=0.01
dev: C1=0.920 C2full=0.660 C2one=0.780 C3scr=0.007 C3adj=0.073 C4=0.013  margin -0.140
test: C1=1.000 C2full=0.787 C2one=0.900 C3scr=0.013 C3adj=0.087 C4=0.040  margin -0.013

### Qwen3-8B sem: all (variant, thr) by dev margin (top 6)
- v2 thr=0.8 dev margin +0.030
- v2 thr=0.7 dev margin +0.007
- v3 thr=0.3 dev margin +0.003
- v3 thr=0.4 dev margin -0.003
- v3 thr=0.2 dev margin -0.007
- v2 thr=0.6 dev margin -0.013
chosen on dev: v2 thr=0.8
dev: C1=0.880 C2full=0.167 C2one=0.440 C3scr=0.000 C3adj=0.053 C4=0.233  margin +0.030
test: C1=0.913 C2full=0.173 C2one=0.433 C3scr=0.000 C3adj=0.047 C4=0.207  margin +0.027

### phi-4 syn: all (variant, thr) by dev margin (top 6)
- v5 thr=0.1 dev margin -0.040
- v2 thr=0.2 dev margin -0.040
- v4 thr=0.1 dev margin -0.060
- v2 thr=0.1 dev margin -0.073
- v5 thr=0.05 dev margin -0.080
- v3 thr=0.1 dev margin -0.087
chosen on dev: v5 thr=0.1
dev: C1=0.953 C2full=0.760 C2one=0.840 C3scr=0.027 C3adj=0.140 C4=0.013  margin -0.040
test: C1=1.000 C2full=0.860 C2one=0.953 C3scr=0.013 C3adj=0.147 C4=0.027  margin -0.047

### phi-4 sem: all (variant, thr) by dev margin (top 6)
- v2 thr=0.9 dev margin +0.053
- v3 thr=0.6 dev margin +0.050
- v2 thr=0.95 dev margin +0.050
- v3 thr=0.5 dev margin +0.047
- v3 thr=0.7 dev margin +0.023
- v3 thr=0.8 dev margin +0.003
chosen on dev: v2 thr=0.9
dev: C1=0.920 C2full=0.147 C2one=0.407 C3scr=0.013 C3adj=0.193 C4=0.413  margin +0.053
test: C1=0.967 C2full=0.253 C2one=0.533 C3scr=0.007 C3adj=0.193 C4=0.407  margin -0.053

## Per-variant at thr .5 (dev | test)
Qwen3-8B/v1/sem: C1 0.95|1.00 C2full 0.37|0.42 C2one 0.68|0.72 C3scr 0.01|0.01 C3adj 0.15|0.13 C4 0.44|0.42
Qwen3-8B/v1/syn: C1 0.91|0.98 C2full 0.47|0.62 C2one 0.71|0.72 C3scr 0.00|0.00 C3adj 0.05|0.03 C4 0.01|0.02
Qwen3-8B/v2/sem: C1 0.91|0.95 C2full 0.21|0.21 C2one 0.51|0.50 C3scr 0.00|0.00 C3adj 0.07|0.07 C4 0.32|0.31
Qwen3-8B/v2/syn: C1 0.89|0.97 C2full 0.53|0.69 C2one 0.72|0.80 C3scr 0.00|0.00 C3adj 0.05|0.05 C4 0.01|0.02
Qwen3-8B/v3/sem: C1 0.82|0.89 C2full 0.13|0.18 C2one 0.33|0.31 C3scr 0.00|0.00 C3adj 0.04|0.05 C4 0.15|0.19
Qwen3-8B/v3/syn: C1 0.91|0.98 C2full 0.53|0.68 C2one 0.70|0.78 C3scr 0.00|0.00 C3adj 0.06|0.05 C4 0.01|0.02
Qwen3-8B/v4/syn: C1 0.91|0.97 C2full 0.53|0.68 C2one 0.72|0.78 C3scr 0.00|0.00 C3adj 0.05|0.03 C4 0.02|0.04
Qwen3-8B/v5/syn: C1 0.89|0.97 C2full 0.55|0.72 C2one 0.71|0.81 C3scr 0.00|0.01 C3adj 0.05|0.05 C4 0.01|0.02
phi-4/v1/sem: C1 0.98|0.99 C2full 0.44|0.55 C2one 0.79|0.87 C3scr 0.05|0.03 C3adj 0.37|0.35 C4 0.63|0.67
phi-4/v1/syn: C1 0.92|0.98 C2full 0.44|0.51 C2one 0.63|0.71 C3scr 0.01|0.01 C3adj 0.05|0.07 C4 0.01|0.02
phi-4/v2/sem: C1 0.96|0.99 C2full 0.31|0.42 C2one 0.66|0.75 C3scr 0.07|0.06 C3adj 0.43|0.45 C4 0.73|0.69
phi-4/v2/syn: C1 0.93|1.00 C2full 0.64|0.79 C2one 0.81|0.91 C3scr 0.02|0.01 C3adj 0.08|0.08 C4 0.01|0.01
phi-4/v3/sem: C1 0.91|0.94 C2full 0.15|0.22 C2one 0.47|0.57 C3scr 0.01|0.00 C3adj 0.15|0.14 C4 0.39|0.41
phi-4/v3/syn: C1 0.87|0.98 C2full 0.60|0.71 C2one 0.71|0.85 C3scr 0.00|0.01 C3adj 0.07|0.07 C4 0.01|0.02
phi-4/v4/syn: C1 0.93|0.99 C2full 0.65|0.75 C2one 0.79|0.91 C3scr 0.02|0.01 C3adj 0.07|0.07 C4 0.01|0.02
phi-4/v5/syn: C1 0.89|0.98 C2full 0.62|0.73 C2one 0.75|0.89 C3scr 0.00|0.00 C3adj 0.05|0.07 C4 0.01|0.01

## 2x2 confusion (test half): syn-pass/sem-pass counts per class (SY=syntax pass, SM=semantics pass)

Qwen3-8B (syn v5 thr 0.01, sem v2 thr 0.8)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 150 | 137 | 13 | 0 | 0 | 1.000 | 0.913 |
| C2full | 150 | 26 | 92 | 0 | 32 | 0.787 | 0.173 |
| C2one | 150 | 65 | 70 | 0 | 15 | 0.900 | 0.433 |
| C3scr | 150 | 0 | 2 | 0 | 148 | 0.013 | 0.000 |
| C3adj | 150 | 6 | 7 | 1 | 136 | 0.087 | 0.047 |
| C4 | 150 | 5 | 1 | 26 | 118 | 0.040 | 0.207 |

phi-4 (syn v5 thr 0.1, sem v2 thr 0.9)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 150 | 145 | 5 | 0 | 0 | 1.000 | 0.967 |
| C2full | 150 | 38 | 91 | 0 | 21 | 0.860 | 0.253 |
| C2one | 150 | 79 | 64 | 1 | 6 | 0.953 | 0.533 |
| C3scr | 150 | 0 | 2 | 1 | 147 | 0.013 | 0.007 |
| C3adj | 150 | 11 | 11 | 18 | 110 | 0.147 | 0.193 |
| C4 | 150 | 3 | 1 | 58 | 88 | 0.027 | 0.407 |

## 2x2 confusion (all half): syn-pass/sem-pass counts per class (SY=syntax pass, SM=semantics pass)

Qwen3-8B (syn v5 thr 0.01, sem v2 thr 0.8)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 300 | 267 | 21 | 2 | 10 | 0.960 | 0.897 |
| C2full | 300 | 50 | 167 | 1 | 82 | 0.723 | 0.170 |
| C2one | 300 | 130 | 122 | 1 | 47 | 0.840 | 0.437 |
| C3scr | 300 | 0 | 3 | 0 | 297 | 0.010 | 0.000 |
| C3adj | 300 | 14 | 10 | 1 | 275 | 0.080 | 0.050 |
| C4 | 300 | 7 | 1 | 59 | 233 | 0.027 | 0.220 |

phi-4 (syn v5 thr 0.1, sem v2 thr 0.9)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 300 | 282 | 11 | 1 | 6 | 0.977 | 0.943 |
| C2full | 300 | 60 | 183 | 0 | 57 | 0.810 | 0.200 |
| C2one | 300 | 140 | 129 | 1 | 30 | 0.897 | 0.470 |
| C3scr | 300 | 2 | 4 | 1 | 293 | 0.020 | 0.010 |
| C3adj | 300 | 24 | 19 | 34 | 223 | 0.143 | 0.193 |
| C4 | 300 | 5 | 1 | 118 | 176 | 0.020 | 0.410 |

### ensemble syn (mean logit of Qwen3-8B:v5, phi-4:v5); chosen on dev thr(logit)=-5.25
dev: C1=0.967 C2full=0.773 C2one=0.887 C3scr=0.007 C3adj=0.127 C4=0.020  margin -0.027
test: C1=1.000 C2full=0.880 C2one=0.953 C3scr=0.020 C3adj=0.127 C4=0.040  margin -0.027

### ensemble sem (mean logit of Qwen3-8B:v2, phi-4:v2); chosen on dev thr(logit)=2.5
dev: C1=0.913 C2full=0.133 C2one=0.387 C3scr=0.000 C3adj=0.060 C4=0.200  margin +0.063
test: C1=0.947 C2full=0.133 C2one=0.440 C3scr=0.000 C3adj=0.060 C4=0.153  margin +0.067

## Ensemble judge 2x2 confusion, test half and all

(test)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 150 | 142 | 8 | 0 | 0 | 1.000 | 0.947 |
| C2full | 150 | 20 | 112 | 0 | 18 | 0.880 | 0.133 |
| C2one | 150 | 66 | 77 | 0 | 7 | 0.953 | 0.440 |
| C3scr | 150 | 0 | 3 | 0 | 147 | 0.020 | 0.000 |
| C3adj | 150 | 9 | 10 | 0 | 131 | 0.127 | 0.060 |
| C4 | 150 | 4 | 2 | 19 | 125 | 0.040 | 0.153 |

(all)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 300 | 279 | 16 | 0 | 5 | 0.983 | 0.930 |
| C2full | 300 | 40 | 208 | 0 | 52 | 0.827 | 0.133 |
| C2one | 300 | 124 | 152 | 0 | 24 | 0.920 | 0.413 |
| C3scr | 300 | 0 | 4 | 0 | 296 | 0.013 | 0.000 |
| C3adj | 300 | 17 | 21 | 1 | 261 | 0.127 | 0.060 |
| C4 | 300 | 6 | 3 | 47 | 244 | 0.030 | 0.177 |

## Both models (AND per axis), test half
| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |
|---|---|---|---|---|---|---|---|
| C1 | 150 | 135 | 15 | 0 | 0 | 1.000 | 0.900 |
| C2full | 150 | 21 | 95 | 0 | 34 | 0.773 | 0.140 |
| C2one | 150 | 55 | 79 | 1 | 15 | 0.893 | 0.373 |
| C3scr | 150 | 0 | 1 | 0 | 149 | 0.007 | 0.000 |
| C3adj | 150 | 5 | 7 | 1 | 137 | 0.080 | 0.040 |
| C4 | 150 | 2 | 0 | 18 | 130 | 0.013 | 0.133 |

## Examples (first five of the test half; P(yes) syn / sem per model)

C1
- then tim's friends came to the park :: Qwen3-8B 1.00/1.00; phi-4 1.00/1.00
- he told them about the big dog :: Qwen3-8B 1.00/1.00; phi-4 1.00/1.00
- at the park sue found a big tree :: Qwen3-8B 1.00/1.00; phi-4 1.00/1.00
- tim picked it up and took it home :: Qwen3-8B 1.00/0.97; phi-4 1.00/0.96
- she was a very good girl :: Qwen3-8B 1.00/1.00; phi-4 1.00/1.00

C2full
- then tim's things came to the man :: Qwen3-8B 1.00/0.41; phi-4 1.00/0.87
- he told them about the proud park :: Qwen3-8B 1.00/0.00; phi-4 1.00/0.50
- at the bird sue thanked a surprised fish :: Qwen3-8B 0.65/0.00; phi-4 0.99/0.16
- tim found it up and found it home :: Qwen3-8B 0.27/0.00; phi-4 0.93/0.02
- she was a very careful box :: Qwen3-8B 1.00/0.00; phi-4 1.00/0.16

C2one
- then tim's friends came to the dog :: Qwen3-8B 1.00/0.22; phi-4 0.80/1.00
- he told them about the big food :: Qwen3-8B 1.00/0.27; phi-4 1.00/0.88
- at the park sue thanked a big tree :: Qwen3-8B 1.00/0.99; phi-4 1.00/0.96
- tim loved it up and took it home :: Qwen3-8B 0.04/0.00; phi-4 1.00/0.09
- she was a very good park :: Qwen3-8B 0.00/0.00; phi-4 0.25/0.04

C3scr
- to tim's came the then friends park :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.02
- he the them about dog told big :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.01
- found a at park sue tree big the :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.00
- it it up took picked and tim home :: Qwen3-8B 0.00/0.00; phi-4 0.01/0.00
- a very girl was she good :: Qwen3-8B 0.00/0.00; phi-4 0.01/0.29

C3adj
- then tim's friends to came the park :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.53
- told he them about the big dog :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.89
- at the park sue a found big tree :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.03
- tim picked it up and took home it :: Qwen3-8B 0.00/0.00; phi-4 0.01/0.27
- she a was very good girl :: Qwen3-8B 0.00/0.01; phi-4 0.00/0.71

C4
- what are that :: Qwen3-8B 0.00/0.00; phi-4 0.00/0.01
- the fish were not happy anymore :: Qwen3-8B 0.97/1.00; phi-4 0.89/1.00
- the cat were not sad anymore :: Qwen3-8B 0.00/0.07; phi-4 0.00/0.98
- what were that :: Qwen3-8B 0.00/0.00; phi-4 0.01/0.03
- she were not scared :: Qwen3-8B 0.00/0.22; phi-4 0.00/0.98
