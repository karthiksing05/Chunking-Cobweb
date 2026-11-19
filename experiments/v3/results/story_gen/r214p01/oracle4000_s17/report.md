Source: {"cond": false, "wreach": 5, "wemit": "leaf", "causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7, "edge": "exact"}; 679 of its 1094 distinct new sentences are R-sem under the judge (they are the oracle's input).

| variant | sentences | changed words per sentence | R-sem | R-syn | R-non | (non with sem_pass) |
|---|---|---|---|---|---|---|
| all-b4 | 2037 | 2.04 | 0.151 | 0.127 | 0.723 | 0.003 |
| all-b6 | 2037 | 1.97 | 0.288 | 0.203 | 0.509 | 0.001 |
| func100-b4 | 2037 | 2.04 | 0.195 | 0.169 | 0.635 | 0.002 |
| func100-b6 | 2037 | 2.03 | 0.251 | 0.203 | 0.546 | 0.003 |
| nocls | 2037 | 2.04 | 0.113 | 0.086 | 0.801 | 0.000 |
| pos | 2037 | 1.99 | 0.242 | 0.217 | 0.541 | 0.003 |

### Samples (5 per variant)

**all-b4**
- suddenly he had an idea  ->  amy he had man other  (syn_p 0.00, sem_p 0.00)
- they had fun and fun playing together  ->  they had fun and fun time together  (syn_p 0.62, sem_p 0.62)
- one day lily found a big tree  ->  spot day lily told a big house  (syn_p 0.03, sem_p 0.00)
- she loved to play outside with his toys  ->  she go to play playing with his excited  (syn_p 0.00, sem_p 0.00)
- lucy was very happy  ->  max was very happy  (syn_p 1.00, sem_p 1.00)

**all-b6**
- the cat was so happy  ->  the end was so happy  (syn_p 0.77, sem_p 0.33)
- sue was happy and scared  ->  sue was happy and scared  (syn_p 1.00, sem_p 0.98)
- they wanted to be friends with the bird  ->  they wanted to be friends with the bird  (syn_p 1.00, sem_p 1.00)
- she saw a big red door  ->  she saw a big outside fish  (syn_p 0.47, sem_p 0.36)
- tim went to the park with her mom  ->  tim liked to the sun with her mom  (syn_p 0.00, sem_p 0.00)

**func100-b4**
- i wanted to be big again  ->  i wanted to be big now  (syn_p 0.98, sem_p 1.00)
- she liked to play with it  ->  she named to play with it  (syn_p 0.00, sem_p 0.00)
- bob was surprised but happy  ->  it's was hard but happy  (syn_p 0.00, sem_p 0.06)
- let's go with the dog  ->  spot get with the toy  (syn_p 0.00, sem_p 0.00)
- i have it for you  ->  i look it knew you  (syn_p 0.00, sem_p 0.00)

**func100-b6**
- they played with the ball together  ->  they played with the anymore together  (syn_p 0.00, sem_p 0.00)
- tom and ben looked at each other  ->  tom and this liked under heard sky  (syn_p 0.00, sem_p 0.00)
- it was sad  ->  it was sad  (syn_p 1.00, sem_p 1.00)
- tom and his friends were surprised but happy  ->  tom and his friends were fast but happy  (syn_p 1.00, sem_p 1.00)
- he had a big room with many toys  ->  he had a big tree with your again  (syn_p 0.00, sem_p 0.00)

**nocls**
- the man and the girl looked at them  ->  the from and the get an special went  (syn_p 0.00, sem_p 0.00)
- he wanted to play with the ball  ->  he wanted to play with the red  (syn_p 0.00, sem_p 0.42)
- lily and tim were very happy and surprised  ->  lily and tim were very happy and sorry  (syn_p 1.00, sem_p 0.89)
- the dog was very excited  ->  the go was very no  (syn_p 0.05, sem_p 0.00)
- the bear said i am sorry  ->  the of said i lots after  (syn_p 0.00, sem_p 0.00)

**pos**
- the bird was not a rock at all  ->  the bird was not a car under all  (syn_p 0.01, sem_p 0.00)
- this is my cake  ->  him is some toys  (syn_p 0.00, sem_p 0.00)
- the house was happy and blue  ->  the room was happy and excited  (syn_p 1.00, sem_p 0.96)
- they were happy to see what was inside  ->  they were happy to get both was after  (syn_p 0.00, sem_p 0.00)
- tom and mia are excited  ->  tom and lucy are good  (syn_p 0.52, sem_p 0.98)
