Source: {"cond": false, "wreach": 5, "wemit": "leaf", "causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7, "edge": "exact"}; 697 of its 1100 distinct new sentences are R-sem under the judge (they are the oracle's input).

| variant | sentences | changed words per sentence | R-sem | R-syn | R-non | (non with sem_pass) |
|---|---|---|---|---|---|---|
| all-b4 | 2091 | 1.94 | 0.195 | 0.149 | 0.656 | 0.002 |
| all-b6 | 2091 | 1.86 | 0.316 | 0.166 | 0.518 | 0.000 |
| func100-b4 | 2091 | 1.94 | 0.215 | 0.130 | 0.656 | 0.001 |
| func100-b6 | 2091 | 1.92 | 0.307 | 0.181 | 0.513 | 0.000 |
| nocls | 2091 | 1.94 | 0.122 | 0.099 | 0.779 | 0.000 |

### Samples (5 per variant)

**all-b4**
- they did not want to share with max  ->  they new not because to shiny with sara  (syn_p 0.00, sem_p 0.00)
- tim and ben were playing in the park  ->  tim and that were every yes the dog  (syn_p 0.00, sem_p 0.00)
- the bird was happy and went away  ->  the bird was happy and looked me  (syn_p 0.02, sem_p 0.20)
- tim felt very sad and scared  ->  tim became very sad and scared  (syn_p 1.00, sem_p 1.00)
- she liked to play all day  ->  she flew to play all day  (syn_p 0.20, sem_p 0.38)

**all-b6**
- they were so happy and excited to go  ->  they were so happy and surprised to see  (syn_p 0.00, sem_p 0.07)
- he looked down and saw his ball  ->  he got up and saw his ball  (syn_p 1.00, sem_p 1.00)
- she was so happy and thanked the mom  ->  she was so happy and of the mom  (syn_p 0.00, sem_p 0.00)
- sara and anna are best friends  ->  amy and sam are my friends  (syn_p 0.99, sem_p 1.00)
- she wanted to help the bird feel better  ->  she wanted to help the bird walked anymore  (syn_p 0.00, sem_p 0.00)

**func100-b4**
- he was so happy he thanked the bird  ->  he was so happy he time the bird  (syn_p 0.00, sem_p 0.00)
- lily was very happy to help his mom  ->  lily was very happy to help his mom  (syn_p 0.97, sem_p 0.97)
- spot was very sad  ->  bob was very sad  (syn_p 1.00, sem_p 1.00)
- he played with the toy all day  ->  he played with the them all day  (syn_p 0.00, sem_p 0.00)
- she loved to play with his ball  ->  she started to play with his ball  (syn_p 1.00, sem_p 1.00)

**func100-b6**
- he saw a big bird on a tree  ->  he saw a big bird there a cake  (syn_p 0.00, sem_p 0.00)
- lily and tom were very happy  ->  lily and tom were very happy  (syn_p 1.00, sem_p 1.00)
- then they saw a big dog named sue  ->  then they saw a big cat down sue  (syn_p 0.02, sem_p 0.00)
- they liked to play in the water  ->  they knew to play yes the ground  (syn_p 0.00, sem_p 0.00)
- one day the cat came into the park  ->  from day the man like upon the ground  (syn_p 0.00, sem_p 0.00)

**nocls**
- he looked around and saw the boy  ->  he food this and saw the at  (syn_p 0.00, sem_p 0.00)
- tim and spot were scared  ->  tim and tree were scared  (syn_p 0.99, sem_p 0.15)
- tom had a big idea  ->  tom had a big hard  (syn_p 0.00, sem_p 0.00)
- tim wanted to be friends with the boy  ->  tim wanted to i'm friends with the everyone  (syn_p 0.00, sem_p 0.00)
- bob was so excited  ->  rock was so me  (syn_p 0.00, sem_p 0.00)
