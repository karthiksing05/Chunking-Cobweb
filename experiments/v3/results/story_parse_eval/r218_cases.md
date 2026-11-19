# r218 case studies (seed 13, n=1000, C1 test sentences with the most distinct trees across the shown cells)


**when they got home something unexpected happened**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `((((when they) (got home)) (something unexpected)) happened)` | 13 |
| W2 | `(((when they) (got home)) ((something unexpected) happened))` | 6 |
| W4 | `(((when they) (got home)) (something (unexpected happened)))` | 15 |
| W4s2 | `(((when they) (got home)) (something (unexpected happened)))` | 25 |
| D1 | `(((when they) ((got home) something)) (unexpected happened))` | 62 |
| D3 | `((when (they (got home))) (something (unexpected happened)))` | 59 |
| D1c | `(((when they) (got home)) ((something unexpected) happened))` | 10 |
| D2c | `(((when they) (got home)) (something (unexpected happened)))` | 29 |
| W4D2 | `(((when they) (got home)) (something (unexpected happened)))` | 15 |
| full | `(((when (they (got home))) something) (unexpected happened))` | 56 |
| W4L | `((((when they) got) home) (something (unexpected happened)))` | 15 |
| D3L | `((when ((they got) home)) (something (unexpected happened)))` | 56 |
| D2cL | `((when ((they got) home)) (something (unexpected happened)))` | 29 |

**soon tim's friends came to the park**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `((soon (tim's friends)) (came (to (the park))))` | 12 |
| W2 | `(((soon tim's) (friends came)) ((to the) park))` | 6 |
| W4 | `((((soon tim's) friends) came) (to (the park)))` | 15 |
| W4s2 | `((soon tim's) ((friends came) (to (the park))))` | 25 |
| D1 | `(((soon tim's) friends) (came (to (the park))))` | 56 |
| D3 | `(((soon tim's) friends) (came (to (the park))))` | 56 |
| D1c | `(((soon tim's) friends) ((came to) (the park)))` | 10 |
| D2c | `(((soon tim's) friends) (came (to (the park))))` | 29 |
| W4D2 | `(((soon tim's) friends) (came (to (the park))))` | 17 |
| full | `(((soon tim's) friends) (came (to (the park))))` | 56 |
| W4L | `((((soon tim's) friends) came) (to (the park)))` | 15 |
| D3L | `((((soon tim's) friends) came) (to (the park)))` | 56 |
| D2cL | `((((soon tim's) friends) came) (to (the park)))` | 29 |

**they were happy and they were friends**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `((((they were) happy) (and (they were))) friends)` | 13 |
| W2 | `(((they were) (happy and)) ((they were) friends))` | 6 |
| W4 | `((((they were) happy) and) (they (were friends)))` | 15 |
| W4s2 | `(((they were) happy) (and (they (were friends))))` | 28 |
| D1 | `(((they were) happy) (and (they (were friends))))` | 56 |
| D3 | `(((they were) happy) (and (they (were friends))))` | 56 |
| D1c | `(((they were) happy) ((and they) (were friends)))` | 10 |
| D2c | `(((they were) happy) (and (they (were friends))))` | 29 |
| W4D2 | `(((they were) happy) (and (they (were friends))))` | 17 |
| full | `(((they were) happy) (and (they (were friends))))` | 56 |
| W4L | `((((they were) happy) and) (they (were friends)))` | 15 |
| D3L | `((((they were) happy) and) ((they were) friends))` | 57 |
| D2cL | `(((they were) happy) (and (they (were friends))))` | 29 |

**when amy found him they both laughed**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `(when ((((amy found) him) (they both)) laughed))` | 15 |
| W2 | `(((when amy) (found him)) ((they both) laughed))` | 6 |
| W4 | `((((when amy) found) him) (they (both laughed)))` | 15 |
| W4s2 | `((when amy) ((found him) (they (both laughed))))` | 25 |
| D1 | `(when (((amy found) him) (they (both laughed))))` | 61 |
| D3 | `(when (((amy found) him) (they (both laughed))))` | 56 |
| D1c | `(((when amy) (found him)) (they (both laughed)))` | 10 |
| D2c | `((((when amy) found) him) (they (both laughed)))` | 29 |
| W4D2 | `((((when amy) found) him) (they (both laughed)))` | 17 |
| full | `(when (((amy found) him) (they (both laughed))))` | 56 |
| W4L | `((((when amy) found) him) (they (both laughed)))` | 15 |
| D3L | `(when (((amy found) him) (they (both laughed))))` | 56 |
| D2cL | `((((when amy) found) him) (they (both laughed)))` | 29 |
