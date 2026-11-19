| arm | n | R-sem | R-legal | R-illegal | = real move |
|---|---|---|---|---|---|
| N0/legal/argmax c+x | 200 | 0.67 | 0.33 | 0.0 | 0.06 |
| N0/legal/argmax c | 200 | 0.65 | 0.35 | 0.0 | 0.055 |
| N0/legal/argmax x | 200 | 0.685 | 0.315 | 0.0 | 0.025 |
| N0/legal/argmin c+x | 200 | 0.72 | 0.28 | 0.0 | 0.045 |
| N0/legal/softmax T=2 | 200 | 0.685 | 0.315 | 0.0 | 0.055 |
| N0/legal/softmax T=8 | 200 | 0.69 | 0.31 | 0.0 | 0.055 |
| N0/legal/uniform | 200 | 0.675 | 0.325 | 0.0 | 0.065 |
| N0/legal/argmax c+x, safe | 200 | 0.73 | 0.27 | 0.0 | 0.095 |
| N300/legal/argmax c+x | 200 | 0.64 | 0.36 | 0.0 | 0.065 |
| N300/legal/argmax c | 200 | 0.72 | 0.28 | 0.0 | 0.055 |
| N300/legal/argmax x | 200 | 0.58 | 0.42 | 0.0 | 0.065 |
| N300/legal/argmin c+x | 200 | 0.64 | 0.36 | 0.0 | 0.035 |
| N300/legal/softmax T=2 | 200 | 0.675 | 0.325 | 0.0 | 0.06 |
| N300/legal/softmax T=8 | 200 | 0.665 | 0.335 | 0.0 | 0.05 |
| N300/legal/uniform | 200 | 0.67 | 0.33 | 0.0 | 0.025 |
| N300/legal/argmax c+x, safe | 200 | 0.725 | 0.275 | 0.0 | 0.095 |
| N1000/legal/argmax c+x | 200 | 0.625 | 0.375 | 0.0 | 0.065 |
| N1000/legal/argmax c | 200 | 0.64 | 0.36 | 0.0 | 0.06 |
| N1000/legal/argmax x | 200 | 0.64 | 0.36 | 0.0 | 0.065 |
| N1000/legal/argmin c+x | 200 | 0.685 | 0.315 | 0.0 | 0.06 |
| N1000/legal/softmax T=2 | 200 | 0.68 | 0.32 | 0.0 | 0.06 |
| N1000/legal/softmax T=8 | 200 | 0.635 | 0.365 | 0.0 | 0.045 |
| N1000/legal/uniform | 200 | 0.655 | 0.345 | 0.0 | 0.04 |
| N1000/legal/argmax c+x, safe | 200 | 0.7 | 0.3 | 0.0 | 0.1 |
| N1000/free/argmax c+x | 200 | 0.125 | 0.105 | 0.77 | 0.0 |
| N1000/free/argmax c | 200 | 0.12 | 0.09 | 0.79 | 0.005 |
| N1000/free/argmax x | 200 | 0.12 | 0.08 | 0.8 | 0.01 |
| N1000/free/argmin c+x | 200 | 0.05 | 0.03 | 0.92 | 0.0 |
| N1000/free/softmax T=2 | 200 | 0.16 | 0.095 | 0.745 | 0.0 |
| N1000/free/softmax T=8 | 200 | 0.195 | 0.075 | 0.73 | 0.005 |
| N1000/free/uniform | 200 | 0.205 | 0.095 | 0.7 | 0.015 |

## Examples

**N0/legal/argmax c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move a2a3 a3: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move e3d3 Qd3: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3d5 Nxd5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5h2 Qh2: R-sem (loss 0); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move f2f3 f3: R-sem (loss 0); real game move h4g5

**N0/legal/argmax c**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move b2b3 b3: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move f1b5 Bb5: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e2d3 Bd3: R-legal (loss 2); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5g3 Qg3: R-sem (loss 0); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3d5 Nd5: R-sem (loss 0); real game move h4g5

**N0/legal/argmax x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c2a4 Qa4: R-sem (loss -3); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move h2h4 h4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move g1h1 Kh1: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5e1 Qe1: R-sem (loss 0); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3c2 Nc2: R-legal (loss 2); real game move h4g5

**N0/legal/argmin c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move e4e5 e5: R-legal (loss 6); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move e3e1 Qe1: R-sem (loss 1); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move a1b1 Rab1: R-legal (loss 2); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5e4 Qe4: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move h1g1 Rg1: R-sem (loss 0); real game move h4g5

**N0/legal/softmax T=2**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move a7b8 Bb8: R-sem (loss 0); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move e3e2 Qe2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e2b5 Bxb5+: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5g5 Qg5: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move f1g2 Bg2: R-sem (loss 0); real game move h4g5

**N0/legal/softmax T=8**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3g3 Rg3: R-legal (loss 4); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e1d1 Red1: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move g2g3 g3: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move f1g2 Bg2: R-sem (loss 0); real game move h4g5

**N0/legal/uniform**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3h3 Rh3: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move d2c4 Nc4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move g2g3 g3: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move a3a4 a4: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move a3a4 a4: R-sem (loss 0); real game move h4g5

**N0/legal/argmax c+x, safe**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c3c4 c4: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move e3d4 Qd4: R-sem (loss 0); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e2d1 Bd1: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move h3h4 h4: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move d3d4 d4: R-sem (loss 0); real game move h4g5

**N300/legal/argmax c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f5 Rf5: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move b3c4 Qc4: R-legal (loss 8); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5c7 Qc7: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e1e2 Ke2: R-sem (loss 0); real game move h4g5

**N300/legal/argmax c**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f5 Rf5: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e2c4 Bc4: R-legal (loss 2); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move d3d4 d4: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move h4h5 h5: R-sem (loss 0); real game move h4g5

**N300/legal/argmax x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c2c1 Qc1: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move b3d5 Qxd5: R-legal (loss 7); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5c7 Qc7: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move f2f3 f3: R-sem (loss 0); real game move h4g5

**N300/legal/argmin c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move g1f1 Kf1: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move d2f3 Nf3: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move g1h2 Kh2: R-sem (loss 0); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move g1h2 Kh2: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3f5 Nf5: R-sem (loss 0); real game move h4g5

**N300/legal/softmax T=2**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move a7c5 Bc5: R-sem (loss 0); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e2c4 Bc4: R-legal (loss 2); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5c7 Qc7: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move g3g4 g4: R-sem (loss 0); real game move h4g5

**N300/legal/softmax T=8**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move a2a4 a4: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move b2d4 Bd4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move b3c4 Qc4: R-legal (loss 8); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move d3d4 d4: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move h1g1 Rg1: R-sem (loss 0); real game move h4g5

**N300/legal/uniform**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c2b1 Qb1: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move b2a3 Ba3: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move g2g4 g4: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5g5 Qg5: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3f5 Nf5: R-sem (loss 0); real game move h4g5

**N300/legal/argmax c+x, safe**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f5 Rf5: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move b3c4 Qc4: R-legal (loss 8); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5c7 Qc7: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e1e2 Ke2: R-sem (loss 0); real game move h4g5

**N1000/legal/argmax c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move g2g4 g4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3b5 Nxb5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5e4 Qe4: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3c4 Nc4: R-sem (loss 0); real game move h4g5

**N1000/legal/argmax c**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3g3 Rg3: R-legal (loss 4); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1c2 Kc2: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3b5 Nxb5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move c3e2 Ne2: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move g3g4 g4: R-sem (loss 0); real game move h4g5

**N1000/legal/argmax x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move b2a1 Ba1: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move b3d1 Qd1: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5e4 Qe4: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3c4 Nc4: R-sem (loss 0); real game move h4g5

**N1000/legal/argmin c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c2d3 Qd3: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move g2g3 g3: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3d5 Nxd5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move g1h2 Kh2: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move d1g4 Qg4: R-legal (loss 6); real game move h4g5

**N1000/legal/softmax T=2**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move g2g4 g4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3b5 Nxb5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5e1 Qe1: R-sem (loss 0); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3c4 Nc4: R-sem (loss 0); real game move h4g5

**N1000/legal/softmax T=8**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move f1c4 Bc4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move a1c1 Rac1: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move f1e1 Rfe1: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move a1a2 Ra2: R-legal (loss 4); real game move h4g5

**N1000/legal/uniform**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move d2c4 Nc4: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move d2c4 Nc4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e2f1 Bf1: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move c3d5 Nxd5: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move d1h5 Qh5: R-legal (loss 8); real game move h4g5

**N1000/legal/argmax c+x, safe**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move g2g4 g4: R-legal (loss 6); real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3b5 Nxb5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5e4 Qe4: R-legal (loss 8); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3c4 Nc4: R-sem (loss 0); real game move h4g5

**N1000/free/argmax c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move a4c6 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e3d1 (illegal): R-illegal; real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move c2c4 (illegal): R-illegal; real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move c1f5 (illegal): R-illegal; real game move h4g5

**N1000/free/argmax c**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c2h7 (illegal): R-illegal; real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move a4c6 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move c3b5 Nxb5: R-sem (loss 1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move c2c4 (illegal): R-illegal; real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move c1g5 (illegal): R-illegal; real game move h4g5

**N1000/free/argmax x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move f3f2 Rf2: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move a4c6 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e3d1 (illegal): R-illegal; real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move h3g6 (illegal): R-illegal; real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e1e7 (illegal): R-illegal; real game move h4g5

**N1000/free/argmin c+x**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move a1d6 (illegal): R-illegal; real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c5g1 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move g1e8 (illegal): R-illegal; real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move d3c8 (illegal): R-illegal; real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move d3c7 (illegal): R-illegal; real game move h4g5

**N1000/free/softmax T=2**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move c2d1 Qd1: R-sem (loss -1); real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move a4c6 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e1h6 (illegal): R-illegal; real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move f1e1 Rfe1: R-sem (loss -1); real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move c1g5 (illegal): R-illegal; real game move h4g5

**N1000/free/softmax T=8**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move b2e5 (illegal): R-illegal; real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move c1h4 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move a1a2 Ra2: R-sem (loss -1); real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move e5c6 (illegal): R-illegal; real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move b2b3 (illegal): R-illegal; real game move h4g5

**N1000/free/uniform**

- `2k1rb1r/Bpp2pp1/3p1nq1/8/4P2p/2P2R2/PPQN2PP/R5K1 w - - 3 16`  move g1a3 (illegal): R-illegal; real game move a1e1
- `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w - - 1 16`  move e3d6 (illegal): R-illegal; real game move e3d4
- `r2qk2r/4npbb/p3p2p/1p1p2p1/3P4/PQN1PN1P/1P1BBPP1/R3R1K1 w kq - 1 16`  move e3a8 (illegal): R-illegal; real game move c3b5
- `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w - - 3 16`  move g1e7 (illegal): R-illegal; real game move c3e2
- `r2q1rk1/2pb1pp1/2np1n2/2b1p1p1/4P2P/Pp1PN1P1/1P3P2/R1BQKB1R w KQ - 0 16`  move e3c4 Nc4: R-sem (loss 0); real game move h4g5
