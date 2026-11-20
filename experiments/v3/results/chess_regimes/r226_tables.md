seeds 13,17; TAU 0.702; boards per cell 600 per seed

### N1000

| cell | R-real | R-legal | R-non | units kept | adjacent kings among illegal |
|---|---|---|---|---|---|
| identity (real board, parse units unchanged) | 0.945 | 0.055 | 0.000 | 0.000 | 0.000 |
| where kc=0 | 0.809 | 0.046 | 0.145 | 0.259 | 0.029 |
| where kc=1 | 0.397 | 0.120 | 0.484 | 0.044 | 0.038 |
| where kc=2 | 0.309 | 0.164 | 0.526 | 0.036 | 0.041 |
| where kc=3 | 0.171 | 0.174 | 0.655 | 0.044 | 0.041 |
| where kc=4 | 0.139 | 0.141 | 0.721 | 0.051 | 0.060 |
| where kc=6 | 0.082 | 0.142 | 0.776 | 0.049 | 0.066 |
| where kc=99 | 0.051 | 0.176 | 0.773 | 0.082 | 0.080 |
| where+what kc=0 | 0.742 | 0.030 | 0.227 | 0.232 | 0.061 |
| where+what kc=1 | 0.161 | 0.040 | 0.799 | 0.019 | 0.072 |
| where+what kc=2 | 0.028 | 0.022 | 0.950 | 0.019 | 0.054 |
| where+what kc=3 | 0.014 | 0.018 | 0.969 | 0.018 | 0.099 |
| where+what kc=4 | 0.008 | 0.018 | 0.973 | 0.019 | 0.107 |
| where+what kc=6 | 0.005 | 0.015 | 0.979 | 0.022 | 0.113 |
| where+what kc=99 | 0.006 | 0.020 | 0.974 | 0.029 | 0.109 |
| list counts=0 | 0.115 | 0.003 | 0.883 | 0.000 | 0.206 |
| list counts=1 | 0.528 | 0.005 | 0.467 | 0.000 | 0.005 |

### N4000

| cell | R-real | R-legal | R-non | units kept | adjacent kings among illegal |
|---|---|---|---|---|---|
| identity (real board, parse units unchanged) | 0.945 | 0.055 | 0.000 | 0.000 | 0.000 |
| where kc=0 | 0.833 | 0.043 | 0.123 | 0.199 | 0.017 |
| where kc=1 | 0.455 | 0.115 | 0.429 | 0.038 | 0.039 |
| where kc=2 | 0.324 | 0.178 | 0.498 | 0.035 | 0.059 |
| where kc=3 | 0.207 | 0.179 | 0.615 | 0.046 | 0.072 |
| where kc=4 | 0.150 | 0.137 | 0.712 | 0.048 | 0.073 |
| where kc=6 | 0.120 | 0.148 | 0.732 | 0.050 | 0.064 |
| where kc=99 | 0.053 | 0.186 | 0.760 | 0.083 | 0.059 |
| where+what kc=0 | 0.802 | 0.018 | 0.180 | 0.177 | 0.077 |
| where+what kc=1 | 0.183 | 0.048 | 0.768 | 0.014 | 0.063 |
| where+what kc=2 | 0.017 | 0.019 | 0.964 | 0.017 | 0.052 |
| where+what kc=3 | 0.014 | 0.010 | 0.976 | 0.014 | 0.084 |
| where+what kc=4 | 0.002 | 0.018 | 0.981 | 0.020 | 0.075 |
| where+what kc=6 | 0.003 | 0.016 | 0.982 | 0.024 | 0.121 |
| where+what kc=99 | 0.005 | 0.013 | 0.982 | 0.026 | 0.128 |
| list counts=0 | 0.100 | 0.002 | 0.899 | 0.000 | 0.230 |
| list counts=1 | 0.492 | 0.006 | 0.502 | 0.000 | 0.007 |

### learning curves (R-real / R-legal / R-non), cells best at N4000 per regime (chosen after the grid)

| N | real: where kc=0 | legal: where kc=99 | non: where+what kc=99 |
|---|---|---|---|
| 500 | 0.809 / 0.053 / 0.138 | 0.060 / 0.152 / 0.786 | 0.000 / 0.018 / 0.982 |
| 1000 | 0.809 / 0.046 / 0.145 | 0.051 / 0.176 / 0.773 | 0.006 / 0.020 / 0.974 |
| 2000 | 0.833 / 0.047 / 0.121 | 0.055 / 0.163 / 0.782 | 0.003 / 0.016 / 0.981 |
| 4000 | 0.833 / 0.043 / 0.123 | 0.053 / 0.186 / 0.760 | 0.005 / 0.013 / 0.982 |

### go rule at any generated cell and N: real max 0.833 (N4000 where kc=0); legal max 0.186 (N4000 where kc=99); non max 0.985 (N2000 where+what kc=6)

### illegal causes, share of ALL boards (N4000, seed 13)

- identity (real board, parse units unchanged): {}
- where kc=0: {'PAWNS_ON_BACKRANK': 0.07, 'OPPOSITE_CHECK': 0.095}
- where kc=1: {'PAWNS_ON_BACKRANK': 0.308, 'OPPOSITE_CHECK': 0.17, 'TOO_MANY_CHECKERS': 0.002, 'IMPOSSIBLE_CHECK': 0.005}
- where kc=2: {'PAWNS_ON_BACKRANK': 0.31, 'OPPOSITE_CHECK': 0.232, 'TOO_MANY_CHECKERS': 0.002, 'IMPOSSIBLE_CHECK': 0.003}
- where kc=3: {'PAWNS_ON_BACKRANK': 0.398, 'OPPOSITE_CHECK': 0.25, 'TOO_MANY_CHECKERS': 0.008, 'IMPOSSIBLE_CHECK': 0.013}
- where kc=4: {'PAWNS_ON_BACKRANK': 0.535, 'TOO_MANY_CHECKERS': 0.018, 'IMPOSSIBLE_CHECK': 0.027, 'OPPOSITE_CHECK': 0.338}
- where kc=6: {'PAWNS_ON_BACKRANK': 0.612, 'OPPOSITE_CHECK': 0.353, 'TOO_MANY_CHECKERS': 0.002, 'IMPOSSIBLE_CHECK': 0.012}
- where kc=99: {'PAWNS_ON_BACKRANK': 0.628, 'IMPOSSIBLE_CHECK': 0.017, 'OPPOSITE_CHECK': 0.383, 'TOO_MANY_CHECKERS': 0.008}
- where+what kc=0: {'TOO_MANY_KINGS': 0.042, 'OPPOSITE_CHECK': 0.067, 'PAWNS_ON_BACKRANK': 0.008, 'NO_WHITE_KING': 0.032, 'NO_BLACK_KING': 0.048, 'TOO_MANY_BLACK_PAWNS': 0.003, 'TOO_MANY_BLACK_PIECES': 0.01}
- where+what kc=1: {'PAWNS_ON_BACKRANK': 0.252, 'TOO_MANY_KINGS': 0.198, 'OPPOSITE_CHECK': 0.117, 'NO_WHITE_KING': 0.17, 'NO_BLACK_KING': 0.247, 'TOO_MANY_WHITE_PAWNS': 0.035, 'TOO_MANY_BLACK_PAWNS': 0.065, 'TOO_MANY_CHECKERS': 0.002, 'IMPOSSIBLE_CHECK': 0.002, 'TOO_MANY_BLACK_PIECES': 0.023, 'TOO_MANY_WHITE_PIECES': 0.008}
- where+what kc=2: {'NO_WHITE_KING': 0.522, 'PAWNS_ON_BACKRANK': 0.54, 'NO_BLACK_KING': 0.4, 'TOO_MANY_KINGS': 0.113, 'TOO_MANY_WHITE_PAWNS': 0.08, 'OPPOSITE_CHECK': 0.107, 'TOO_MANY_WHITE_PIECES': 0.008, 'TOO_MANY_BLACK_PIECES': 0.03, 'TOO_MANY_BLACK_PAWNS': 0.065, 'TOO_MANY_CHECKERS': 0.003, 'IMPOSSIBLE_CHECK': 0.003}
- where+what kc=3: {'NO_WHITE_KING': 0.533, 'NO_BLACK_KING': 0.408, 'TOO_MANY_KINGS': 0.153, 'PAWNS_ON_BACKRANK': 0.612, 'TOO_MANY_CHECKERS': 0.007, 'IMPOSSIBLE_CHECK': 0.007, 'TOO_MANY_WHITE_PAWNS': 0.105, 'TOO_MANY_WHITE_PIECES': 0.018, 'OPPOSITE_CHECK': 0.125, 'TOO_MANY_BLACK_PAWNS': 0.137, 'TOO_MANY_BLACK_PIECES': 0.065}
- where+what kc=4: {'NO_WHITE_KING': 0.522, 'NO_BLACK_KING': 0.407, 'TOO_MANY_BLACK_PAWNS': 0.253, 'PAWNS_ON_BACKRANK': 0.662, 'TOO_MANY_KINGS': 0.17, 'TOO_MANY_WHITE_PAWNS': 0.048, 'OPPOSITE_CHECK': 0.167, 'TOO_MANY_BLACK_PIECES': 0.107, 'TOO_MANY_WHITE_PIECES': 0.018, 'TOO_MANY_CHECKERS': 0.003, 'IMPOSSIBLE_CHECK': 0.005}
- where+what kc=6: {'NO_BLACK_KING': 0.355, 'PAWNS_ON_BACKRANK': 0.695, 'TOO_MANY_BLACK_PAWNS': 0.127, 'OPPOSITE_CHECK': 0.205, 'TOO_MANY_KINGS': 0.268, 'TOO_MANY_WHITE_PAWNS': 0.107, 'NO_WHITE_KING': 0.442, 'TOO_MANY_BLACK_PIECES': 0.108, 'TOO_MANY_WHITE_PIECES': 0.057, 'TOO_MANY_CHECKERS': 0.008, 'IMPOSSIBLE_CHECK': 0.018}
- where+what kc=99: {'TOO_MANY_KINGS': 0.268, 'PAWNS_ON_BACKRANK': 0.738, 'NO_WHITE_KING': 0.412, 'NO_BLACK_KING': 0.36, 'TOO_MANY_BLACK_PAWNS': 0.188, 'TOO_MANY_BLACK_PIECES': 0.095, 'OPPOSITE_CHECK': 0.227, 'TOO_MANY_CHECKERS': 0.018, 'IMPOSSIBLE_CHECK': 0.025, 'TOO_MANY_WHITE_PAWNS': 0.112, 'TOO_MANY_WHITE_PIECES': 0.055}
- list counts=0: {'TOO_MANY_KINGS': 0.34, 'NO_WHITE_KING': 0.32, 'TOO_MANY_WHITE_PAWNS': 0.117, 'TOO_MANY_BLACK_PAWNS': 0.098, 'TOO_MANY_BLACK_PIECES': 0.058, 'NO_BLACK_KING': 0.32, 'TOO_MANY_WHITE_PIECES': 0.045, 'IMPOSSIBLE_CHECK': 0.005, 'OPPOSITE_CHECK': 0.058, 'TOO_MANY_CHECKERS': 0.002}
- list counts=1: {'OPPOSITE_CHECK': 0.072, 'NO_BLACK_KING': 0.27, 'NO_WHITE_KING': 0.253, 'TOO_MANY_CHECKERS': 0.002, 'IMPOSSIBLE_CHECK': 0.002}

### showcase boards (N4000, seed 13): board field of the FEN, ranks 8 to 1

- where kc=0 [real] 1r4k1/2pR1pp1/r1p1n1p1/p1B2P2/2P3P1/7P/1PP5/2K4R
- where kc=0 [real] 3q1rk1/1p2n1pp/2pb4/p5r1/2N1PQ1p/2P5/PP3PPP/R1B2RK1
- where kc=0 [real] 3q1rk1/1p2n1pp/2pb4/p5r1/2N1PQ1p/2P5/PP3PPP/R1B2RK1
- where kc=0 [legal] 1r4k1/2p3p1/r1p1n3/p4P2/2P3P1/2R1p2P/1PP2p2/1BK4R
- where kc=0 [legal] r3k2r/pp3pp1/p2p1q2/1pp5/3n4/P1bP2P1/1PPBP3/R2QK1R1
- where kc=0 [legal] rn2k2r/pp3ppp/5q2/4p3/5pp1/P1bP2P1/1PPBP3/R2QK1R1
- where kc=0 [non] r1p1r3/pp1p2k1/p6p/2Q3P1/3P4/2P4P/PP3P2/6K1
- where kc=0 [non] r1p1r1k1/pp1p3p/p7/2Q3P1/3P4/2P4P/PP3P2/6K1
- where kc=0 [non] r3rpk1/pppn2p1/5q2/4N2p/3P1P2/1Q4Pb/1P1P1P1P/2R1R1KB
- where kc=99 [real] 8/1k6/p1p1p3/8/3P4/P1P1R3/4K3/6r1
- where kc=99 [real] r2q1rk1/ppp1bpp1/1Pnpbn1p/1P2B3/1P1QP3/1R1KPN2/B5PP/7R
- where kc=99 [real] 3q1rk1/1p1Bbpp1/P1Qpbn1p/R2Br3/2P1p1p1/2P1PNn1/5PPP/4K2R
- where kc=99 [legal] 2r5/1Q1pqrk1/5npp/2ppb1P1/2N1P3/1pPP4/2PPP3/R1BK1R2
- where kc=99 [legal] 6k1/r3r2p/pp1p4/1PP1pPP1/1pP4K/1P1Q2P1/8/8
- where kc=99 [legal] 8/2rp1kr1/PPpppPp1/1p2P1K1/1P1Q4/P4P2/8/8
- where kc=99 [non] 8/4P3/r1P2kP1/4RppP/PP1n3p/1K1B2pR/4r1p1/4p3
- where kc=99 [non] 1rR1p1k1/5pp1/1B2n3/5pP1/P2rPp2/3p1P2/2PP4/3K1R2
- where kc=99 [non] 1rpq4/Q1p1n3/1ppb4/8/rkN3P1/1pp1P3/2PP1PPP/R1B2RK1
- where+what kc=99 [real] 1k6/4P3/3P4/3pp3/1r6/4P3/6PP/6K1
- where+what kc=99 [real] 5k2/4r1p1/4np2/8/4K3/5QP1/5PB1/2R3R1
- where+what kc=99 [real] 8/8/2p5/r1bpprk1/P2pn1p1/8/rPP5/1K6
- where+what kc=99 [legal] 8/2r3kb/8/1pN2p1n/PPPPp2p/2P5/4RP2/5K2
- where+what kc=99 [legal] 5r1k/3P1p2/PR6/2P3p1/1rr2PPP/3R2K1/2p5/2r5
- where+what kc=99 [legal] 8/2qnQ1n1/1k1bb3/1pp3p1/pp6/P2PpP2/2PpP3/3K4
- where+what kc=99 [non] 2P2N2/PP6/1K1R1B1R/3p4/3p4/8/1P1P1k2/K2R1b2
- where+what kc=99 [non] 2pp3p/4p3/7b/2Q3n1/2Nq4/2R1B3/2Pn1p2/N3p3
- where+what kc=99 [non] 6p1/7p/2rkQB1P/1r1PK1r1/1pnp2pN/1pppppP1/1pp1Pp2/8
- where kc=99 [real] 8/1k6/p1p1p3/8/3P4/P1P1R3/4K3/6r1
- where kc=99 [real] r2q1rk1/ppp1bpp1/1Pnpbn1p/1P2B3/1P1QP3/1R1KPN2/B5PP/7R
- where kc=99 [real] 3q1rk1/1p1Bbpp1/P1Qpbn1p/R2Br3/2P1p1p1/2P1PNn1/5PPP/4K2R
- where kc=99 [legal] 2r5/1Q1pqrk1/5npp/2ppb1P1/2N1P3/1pPP4/2PPP3/R1BK1R2
- where kc=99 [legal] 6k1/r3r2p/pp1p4/1PP1pPP1/1pP4K/1P1Q2P1/8/8
- where kc=99 [legal] 8/2rp1kr1/PPpppPp1/1p2P1K1/1P1Q4/P4P2/8/8
- where kc=99 [non] 8/4P3/r1P2kP1/4RppP/PP1n3p/1K1B2pR/4r1p1/4p3
- where kc=99 [non] 1rR1p1k1/5pp1/1B2n3/5pP1/P2rPp2/3p1P2/2PP4/3K1R2
- where kc=99 [non] 1rpq4/Q1p1n3/1ppb4/8/rkN3P1/1pp1P3/2PP1PPP/R1B2RK1
- where+what kc=99 [real] 1k6/4P3/3P4/3pp3/1r6/4P3/6PP/6K1
- where+what kc=99 [real] 5k2/4r1p1/4np2/8/4K3/5QP1/5PB1/2R3R1
- where+what kc=99 [real] 8/8/2p5/r1bpprk1/P2pn1p1/8/rPP5/1K6
- where+what kc=99 [legal] 8/2r3kb/8/1pN2p1n/PPPPp2p/2P5/4RP2/5K2
- where+what kc=99 [legal] 5r1k/3P1p2/PR6/2P3p1/1rr2PPP/3R2K1/2p5/2r5
- where+what kc=99 [legal] 8/2qnQ1n1/1k1bb3/1pp3p1/pp6/P2PpP2/2PpP3/3K4
- where+what kc=99 [non] 2P2N2/PP6/1K1R1B1R/3p4/3p4/8/1P1P1k2/K2R1b2
- where+what kc=99 [non] 2pp3p/4p3/7b/2Q3n1/2Nq4/2R1B3/2Pn1p2/N3p3
- where+what kc=99 [non] 6p1/7p/2rkQB1P/1r1PK1r1/1pnp2pN/1pppppP1/1pp1Pp2/8