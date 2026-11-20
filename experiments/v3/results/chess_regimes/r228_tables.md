### (a) bits per position after 4,000 positions (mean over seeds; lower is better; model cost not charged)

| set | seeds | full recipe (best level) | concepts off (leaf-only level 0/0) | chunks off (max size 1) | piece-square list | BPE over squares | BPE over shapes |
|---|---|---|---|---|---|---|---|
| control | [13, 17, 19] | 129.2 | 129.9 | 183.3 | 181.9 | 109.6 | 111.4 |
| family | [13, 17, 19] | 137.5 | 138.2 | 191.9 | 190.4 | 123.7 | 120.4 |
| wing | [13, 17, 19] | 153.4 | 153.8 | 200.7 | 198.3 | 146.5 | 142.7 |
| shift | [13, 17, 19] | 151.5 | 152.1 | 210.7 | 208.9 | 200.4 | 149.7 |

per seed (full / leaf / off): control: {'full': [130.8, 128.9, 127.8], 'leaf': [131.7, 129.5, 128.5], 'off': [183.2, 183.2, 183.5]}; family: {'full': [139.6, 138.2, 134.6], 'leaf': [140.4, 138.9, 135.2], 'off': [192.5, 192.1, 191.2]}; wing: {'full': [154.0, 154.0, 152.2], 'leaf': [154.3, 154.5, 152.7], 'off': [200.4, 200.8, 200.8]}; shift: {'full': [153.7, 150.1, 150.6], 'leaf': [154.3, 151.0, 151.1], 'off': [210.8, 211.1, 210.1]}

### (b) Chase & Simon trial-1 correct pieces at 4,000 positions (mean over seeds; 10 subjects each)

| arm | seeds | K | middle | end | random middle | random end |
|---|---|---|---|---|---|---|
| full recipe (cs73x) | [13, 17, 19] | K=7 | 12.91 | 7.38 | 5.93 | 6.12 |
| full recipe (cs73x) | [13, 17, 19] | K=4 | 9.53 | 4.47 | 3.15 | 3.23 |
|  stored chunks at 4000 (per seed) | | | [14849, 13951, 14820] | | | |
| concepts off (leaf-only reads) | [13, 17, 19] | K=7 | 0.00 | 0.00 | 0.00 | 0.00 |
| concepts off (leaf-only reads) | [13, 17, 19] | K=4 | 0.00 | 0.00 | 0.00 | 0.00 |
|  stored chunks at 4000 (per seed) | | | [14849, 13951, 14820] | | | |
| chunks off (max size 1) | [13, 17, 19] | K=7 | 6.98 | 6.89 | 6.37 | 6.27 |
| chunks off (max size 1) | [13, 17, 19] | K=4 | 3.99 | 3.94 | 3.47 | 3.40 |
|  stored chunks at 4000 (per seed) | | | [0, 0, 0] | | | |