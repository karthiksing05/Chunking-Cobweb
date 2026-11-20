seeds 13,17; TAU 0.702; boards per cell 600 per seed (300 held-out positions x 2 draws)

### N1000 (parsed chunks per position: 5.48)

| cell | R-real | R-legal | R-non | chunks kept |
|---|---|---|---|---|
| identity (real board, parse units unchanged) | 0.945 | 0.055 | 0.000 | 0.000 |
| keep kc=0 | 0.672 | 0.036 | 0.291 | 0.261 |
| keep kc=1 | 0.179 | 0.058 | 0.762 | 0.020 |
| keep kc=2 | 0.037 | 0.021 | 0.943 | 0.015 |
| keep kc=3 | 0.021 | 0.018 | 0.962 | 0.013 |
| keep kc=4 | 0.010 | 0.021 | 0.969 | 0.013 |
| keep kc=6 | 0.010 | 0.025 | 0.966 | 0.017 |
| keep kc=99 | 0.007 | 0.021 | 0.972 | 0.013 |
| resample kc=0 | 0.682 | 0.042 | 0.275 | 0.260 |
| resample kc=1 | 0.188 | 0.027 | 0.786 | 0.017 |
| resample kc=2 | 0.032 | 0.017 | 0.952 | 0.013 |
| resample kc=3 | 0.021 | 0.013 | 0.966 | 0.011 |
| resample kc=4 | 0.019 | 0.014 | 0.968 | 0.014 |
| resample kc=6 | 0.007 | 0.025 | 0.968 | 0.017 |
| resample kc=99 | 0.005 | 0.017 | 0.978 | 0.015 |

### N4000 (parsed chunks per position: 5.49)

| cell | R-real | R-legal | R-non | chunks kept |
|---|---|---|---|---|
| identity (real board, parse units unchanged) | 0.945 | 0.055 | 0.000 | 0.000 |
| keep kc=0 | 0.762 | 0.051 | 0.188 | 0.179 |
| keep kc=1 | 0.228 | 0.049 | 0.722 | 0.010 |
| keep kc=2 | 0.043 | 0.019 | 0.938 | 0.009 |
| keep kc=3 | 0.020 | 0.019 | 0.960 | 0.009 |
| keep kc=4 | 0.018 | 0.018 | 0.965 | 0.015 |
| keep kc=6 | 0.005 | 0.015 | 0.980 | 0.013 |
| keep kc=99 | 0.007 | 0.018 | 0.974 | 0.012 |
| resample kc=0 | 0.762 | 0.050 | 0.188 | 0.178 |
| resample kc=1 | 0.223 | 0.048 | 0.730 | 0.010 |
| resample kc=2 | 0.055 | 0.019 | 0.926 | 0.010 |
| resample kc=3 | 0.009 | 0.012 | 0.979 | 0.008 |
| resample kc=4 | 0.009 | 0.010 | 0.980 | 0.011 |
| resample kc=6 | 0.009 | 0.008 | 0.984 | 0.012 |
| resample kc=99 | 0.005 | 0.017 | 0.980 | 0.014 |

### learning curves (R-real / R-legal / R-non), cells best at N4000 per regime (chosen after the grid)

| N | real: resample kc=0 | legal: keep kc=0 | non: resample kc=6 |
|---|---|---|---|
| 500 | 0.667 / 0.041 / 0.291 | 0.660 / 0.039 / 0.301 | 0.009 / 0.019 / 0.972 |
| 1000 | 0.682 / 0.042 / 0.275 | 0.672 / 0.036 / 0.291 | 0.007 / 0.025 / 0.968 |
| 2000 | 0.790 / 0.048 / 0.161 | 0.789 / 0.046 / 0.164 | 0.007 / 0.018 / 0.976 |
| 4000 | 0.762 / 0.050 / 0.188 | 0.762 / 0.051 / 0.188 | 0.009 / 0.008 / 0.984 |

### go rule at any generated cell and N: real max 0.790 (N2000 resample kc=0); legal max 0.058 (N1000 keep kc=1); non max 0.984 (N500 resample kc=99)

### examples (N4000, seed 13)

- keep kc=0 [real] 1r4k1/2pR1pp1/r1p1n1p1/p1B1P3/8/4P2P/PP4P1/1K5R
- keep kc=0 [real] 1r4k1/2pR1pp1/r1p1n1p1/p1B1P3/8/4P2P/PP4P1/1K5R
- keep kc=0 [legal] r3k2r/ppp1ppb1/2p1b3/3p4/1nPP4/3P1N2/1PPN1PPP/R1BK1B1R
- keep kc=0 [legal] r3k2r/ppp1ppb1/2p1b3/3p4/1nPP4/3P1N2/1PPN1PPP/R1BK1B1R
- keep kc=0 [non] r3r1k1/pp4pp/3n1qp1/5p2/3P1P2/1Q2N1Pb/PP3P1P/2R1R1KB
- keep kc=0 [non] r3r1k1/pp4pp/3n1qp1/5p2/3P1P2/1Q2N1Pb/PP3P1P/2R1R1KB
- keep kc=2 [real] r2q4/ppp3rk/3p1P2/4p3/3P1bP1/2P5/4PP1P/6K1
- keep kc=2 [real] 5rk1/3pqb2/2p3p1/1p1Pp3/p1ppP2P/2P5/PP2P1K1/8
- keep kc=2 [legal] 1n6/r1kbqp2/2p1p3/3pPP1B/3P2PQ/3B1N1P/8/R4RK1
- keep kc=2 [legal] 1rr2k2/R3p1p1/n4p2/rbp5/1K6/1P6/7P/7b
- keep kc=2 [non] nbq5/b7/p2Qrk2/PPn5/3P4/5P2/6P1/1R1R3P
- keep kc=2 [non] 8/8/rb1krk2/p1q5/8/6P1/5P2/1KP4R
- keep kc=99 [real] 8/8/1kpP4/2N1P3/PPp5/3np3/5KP1/8
- keep kc=99 [real] r2qk3/8/4bn2/8/8/7p/5PPP/6K1
- keep kc=99 [legal] 8/8/2P3r1/PP4pp/1Kpp4/1pr1k3/p3pp2/8
- keep kc=99 [legal] 8/k7/1p3PPP/6K1/p2ppp2/1p1r4/4bp2/5n2
- keep kc=99 [non] 2P5/PP3P2/1rP1K1P1/P1pp4/r7/7P/p5PP/1p5K
- keep kc=99 [non] 1rk5/5b2/b2prk2/n1p1p3/8/3r2k1/6p1/1r1k3p
- resample kc=2 [real] 8/2r3np/8/2k2N2/5PP1/4K1P1/8/3q4
- resample kc=2 [real] 4rr1k/3q1rpp/3pn1p1/2b1pp2/1P3RQP/P5KB/1P1R1P2/1B4R1
- resample kc=2 [legal] 1n2r3/1pk1qpp1/2bp1b1p/4q3/5n1B/3BpN2/RN1PPP1P/N3K2R
- resample kc=2 [legal] 4rq2/5n1p/3p4/2k1p3/1R6/4p3/1QP1P2P/6K1
- resample kc=2 [non] 5q2/6b1/1b2pp2/p1qp4/8/3PPP2/2P2P2/1P5P
- resample kc=2 [non] r2rr3/br3qr1/2q1n3/n1n5/7P/1P4P1/7P/1PN4K
- resample kc=99 [real] 7r/5b2/3k4/5p1p/8/5P2/4PKP1/6b1
- resample kc=99 [legal] 1rr4r/3rk3/n1pq4/5b2/7b/2pp1K2/2p3PP/RNB2RR1
- resample kc=99 [legal] 4rk2/b3p3/2pPP3/1pBnRr2/3Pp1n1/n1q4b/3br3/4n1K1
- resample kc=99 [non] 8/2qpk3/rn2p3/p1b5/4B3/5P2/2p3bk/1P5p
- resample kc=99 [non] 8/5P2/1N1NBPP1/P1PP4/3rk3/4b2P/6P1/1p5R

### illegal causes (N4000, seed 13)

- identity (real board, parse units unchanged): {}
- keep kc=0: {'OPPOSITE_CHECK': 0.04, 'NO_WHITE_KING': 0.027, 'TOO_MANY_KINGS': 0.023, 'PAWNS_ON_BACKRANK': 0.033, 'NO_BLACK_KING': 0.057, 'TOO_MANY_BLACK_PIECES': 0.007}
- keep kc=1: {'PAWNS_ON_BACKRANK': 0.258, 'NO_WHITE_KING': 0.148, 'TOO_MANY_KINGS': 0.202, 'OPPOSITE_CHECK': 0.088, 'NO_BLACK_KING': 0.21, 'TOO_MANY_BLACK_PAWNS': 0.027, 'TOO_MANY_BLACK_PIECES': 0.01, 'TOO_MANY_WHITE_PAWNS': 0.027, 'IMPOSSIBLE_CHECK': 0.003, 'TOO_MANY_WHITE_PIECES': 0.008, 'TOO_MANY_CHECKERS': 0.002}
- keep kc=2: {'NO_WHITE_KING': 0.477, 'PAWNS_ON_BACKRANK': 0.583, 'TOO_MANY_KINGS': 0.113, 'NO_BLACK_KING': 0.373, 'TOO_MANY_WHITE_PAWNS': 0.067, 'TOO_MANY_BLACK_PAWNS': 0.047, 'OPPOSITE_CHECK': 0.088, 'TOO_MANY_BLACK_PIECES': 0.012, 'TOO_MANY_WHITE_PIECES': 0.012}
- keep kc=3: {'TOO_MANY_KINGS': 0.117, 'NO_WHITE_KING': 0.523, 'PAWNS_ON_BACKRANK': 0.667, 'OPPOSITE_CHECK': 0.117, 'NO_BLACK_KING': 0.405, 'TOO_MANY_BLACK_PAWNS': 0.112, 'TOO_MANY_WHITE_PAWNS': 0.092, 'TOO_MANY_WHITE_PIECES': 0.017, 'TOO_MANY_BLACK_PIECES': 0.035, 'TOO_MANY_CHECKERS': 0.002, 'IMPOSSIBLE_CHECK': 0.002}
- keep kc=4: {'NO_WHITE_KING': 0.467, 'NO_BLACK_KING': 0.417, 'PAWNS_ON_BACKRANK': 0.693, 'TOO_MANY_KINGS': 0.175, 'TOO_MANY_BLACK_PIECES': 0.087, 'TOO_MANY_BLACK_PAWNS': 0.212, 'TOO_MANY_WHITE_PAWNS': 0.068, 'OPPOSITE_CHECK': 0.138, 'TOO_MANY_WHITE_PIECES': 0.018, 'IMPOSSIBLE_CHECK': 0.01, 'TOO_MANY_CHECKERS': 0.007}
- keep kc=6: {'NO_BLACK_KING': 0.338, 'PAWNS_ON_BACKRANK': 0.715, 'NO_WHITE_KING': 0.415, 'TOO_MANY_BLACK_PIECES': 0.082, 'TOO_MANY_KINGS': 0.282, 'TOO_MANY_WHITE_PAWNS': 0.122, 'TOO_MANY_BLACK_PAWNS': 0.122, 'IMPOSSIBLE_CHECK': 0.015, 'OPPOSITE_CHECK': 0.198, 'TOO_MANY_WHITE_PIECES': 0.063, 'TOO_MANY_CHECKERS': 0.005}
- keep kc=99: {'NO_BLACK_KING': 0.332, 'TOO_MANY_WHITE_PAWNS': 0.128, 'PAWNS_ON_BACKRANK': 0.713, 'NO_WHITE_KING': 0.423, 'TOO_MANY_KINGS': 0.287, 'OPPOSITE_CHECK': 0.205, 'TOO_MANY_BLACK_PAWNS': 0.165, 'TOO_MANY_BLACK_PIECES': 0.087, 'TOO_MANY_CHECKERS': 0.005, 'IMPOSSIBLE_CHECK': 0.008, 'TOO_MANY_WHITE_PIECES': 0.075}
- resample kc=0: {'OPPOSITE_CHECK': 0.042, 'TOO_MANY_KINGS': 0.025, 'NO_WHITE_KING': 0.027, 'PAWNS_ON_BACKRANK': 0.035, 'NO_BLACK_KING': 0.06, 'TOO_MANY_BLACK_PIECES': 0.007}
- resample kc=1: {'NO_BLACK_KING': 0.25, 'PAWNS_ON_BACKRANK': 0.253, 'NO_WHITE_KING': 0.148, 'TOO_MANY_BLACK_PAWNS': 0.043, 'TOO_MANY_KINGS': 0.192, 'OPPOSITE_CHECK': 0.1, 'TOO_MANY_WHITE_PAWNS': 0.022, 'TOO_MANY_BLACK_PIECES': 0.015, 'TOO_MANY_WHITE_PIECES': 0.008}
- resample kc=2: {'NO_WHITE_KING': 0.487, 'NO_BLACK_KING': 0.368, 'PAWNS_ON_BACKRANK': 0.588, 'TOO_MANY_WHITE_PAWNS': 0.077, 'TOO_MANY_KINGS': 0.143, 'OPPOSITE_CHECK': 0.078, 'TOO_MANY_BLACK_PAWNS': 0.063, 'TOO_MANY_BLACK_PIECES': 0.013, 'TOO_MANY_WHITE_PIECES': 0.012, 'IMPOSSIBLE_CHECK': 0.002}
- resample kc=3: {'NO_BLACK_KING': 0.42, 'PAWNS_ON_BACKRANK': 0.677, 'TOO_MANY_KINGS': 0.145, 'TOO_MANY_WHITE_PAWNS': 0.075, 'NO_WHITE_KING': 0.545, 'TOO_MANY_WHITE_PIECES': 0.023, 'TOO_MANY_BLACK_PAWNS': 0.127, 'OPPOSITE_CHECK': 0.117, 'TOO_MANY_BLACK_PIECES': 0.035, 'IMPOSSIBLE_CHECK': 0.002}
- resample kc=4: {'NO_BLACK_KING': 0.433, 'PAWNS_ON_BACKRANK': 0.707, 'TOO_MANY_KINGS': 0.173, 'OPPOSITE_CHECK': 0.145, 'NO_WHITE_KING': 0.49, 'TOO_MANY_BLACK_PAWNS': 0.223, 'TOO_MANY_BLACK_PIECES': 0.073, 'TOO_MANY_WHITE_PAWNS': 0.073, 'TOO_MANY_WHITE_PIECES': 0.033, 'IMPOSSIBLE_CHECK': 0.008, 'TOO_MANY_CHECKERS': 0.005}
- resample kc=6: {'NO_WHITE_KING': 0.417, 'PAWNS_ON_BACKRANK': 0.702, 'OPPOSITE_CHECK': 0.202, 'TOO_MANY_KINGS': 0.293, 'NO_BLACK_KING': 0.353, 'TOO_MANY_BLACK_PAWNS': 0.133, 'TOO_MANY_WHITE_PAWNS': 0.115, 'TOO_MANY_WHITE_PIECES': 0.04, 'TOO_MANY_CHECKERS': 0.005, 'IMPOSSIBLE_CHECK': 0.007, 'TOO_MANY_BLACK_PIECES': 0.098}
- resample kc=99: {'NO_WHITE_KING': 0.388, 'PAWNS_ON_BACKRANK': 0.753, 'OPPOSITE_CHECK': 0.227, 'TOO_MANY_BLACK_PAWNS': 0.178, 'TOO_MANY_WHITE_PAWNS': 0.1, 'TOO_MANY_WHITE_PIECES': 0.047, 'TOO_MANY_KINGS': 0.278, 'NO_BLACK_KING': 0.353, 'TOO_MANY_BLACK_PIECES': 0.088, 'TOO_MANY_CHECKERS': 0.005, 'IMPOSSIBLE_CHECK': 0.008}