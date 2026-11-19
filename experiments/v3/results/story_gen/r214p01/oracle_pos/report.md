Source: {"cond": false, "wreach": 7, "wemit": "leaf", "wemitk": 0, "causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7, "edge": "exact"}; 468 of its 743 distinct new sentences are R-sem under the judge (they are the oracle's input).

| variant | sentences | changed words per sentence | R-sem | R-syn | R-non | (non with sem_pass) |
|---|---|---|---|---|---|---|
| all-b4 | 1404 | 1.78 | 0.251 | 0.160 | 0.590 | 0.003 |
| all-b6 | 1404 | 1.70 | 0.381 | 0.208 | 0.411 | 0.006 |
| func100-b4 | 1404 | 1.78 | 0.251 | 0.153 | 0.595 | 0.005 |
| func100-b6 | 1404 | 1.71 | 0.348 | 0.165 | 0.487 | 0.009 |
| nocls | 1404 | 1.78 | 0.167 | 0.078 | 0.755 | 0.001 |
| pos | 1404 | 1.72 | 0.310 | 0.199 | 0.491 | 0.004 |

### Samples (5 per variant)

**all-b4**
- they loved to play at the water  ->  they started to play loved the house  (syn_p 0.00, sem_p 0.00)
- she wanted to play with her ball  ->  she wanted to play with her ball  (syn_p 1.00, sem_p 1.00)
- sue was very sad  ->  sue was very sad  (syn_p 1.00, sem_p 1.00)
- one day she saw a big door  ->  lucy day she each a big back  (syn_p 0.00, sem_p 0.00)
- tom and tom were very surprised  ->  tom and tom were very shiny  (syn_p 0.99, sem_p 0.29)

**all-b6**
- they started to play too  ->  they want to play too  (syn_p 0.99, sem_p 1.00)
- the cat said yes  ->  the bird said no  (syn_p 1.00, sem_p 1.00)
- lily and sue were sad  ->  lily and sue were sad  (syn_p 1.00, sem_p 1.00)
- sue and tom were very sad  ->  sue and tom were very sad  (syn_p 1.00, sem_p 1.00)
- he did not get it  ->  he about not be it  (syn_p 0.00, sem_p 0.00)

**func100-b4**
- lily was very small  ->  lily was very says  (syn_p 0.00, sem_p 0.00)
- lily and lily looked at each other  ->  lily and lily laughed little old house  (syn_p 0.00, sem_p 0.00)
- lily was very happy and said yes  ->  lily was very happy and said ben  (syn_p 0.03, sem_p 0.00)
- he saw a new friend  ->  he am a felt toy  (syn_p 0.00, sem_p 0.00)
- we want to go now  ->  yes old to go anna  (syn_p 0.00, sem_p 0.00)

**func100-b6**
- it is not happy  ->  it is not happy  (syn_p 1.00, sem_p 0.98)
- mia and bob played together all day  ->  we and thanked played together all day  (syn_p 0.00, sem_p 0.01)
- tom wanted to play too  ->  tom wanted to play too  (syn_p 1.00, sem_p 1.00)
- they wanted to help the cat  ->  they wanted to run the bird  (syn_p 0.00, sem_p 0.00)
- tim and tom were surprised  ->  tim and tom were scared  (syn_p 1.00, sem_p 1.00)

**nocls**
- tom was so happy and excited  ->  tom was so happy and after  (syn_p 0.00, sem_p 0.01)
- he had a big dog named spot  ->  he had a big made amy girl  (syn_p 0.00, sem_p 0.00)
- lily felt so happy and excited  ->  lily every so happy and under  (syn_p 0.00, sem_p 0.00)
- tim was very happy to help him  ->  tim was very happy to come red  (syn_p 0.00, sem_p 0.00)
- tim and anna are best friends  ->  tim and other are man friends  (syn_p 0.00, sem_p 0.00)

**pos**
- he is good  ->  he is blue  (syn_p 1.00, sem_p 0.98)
- i had a very good friend  ->  i had a very excited boy  (syn_p 0.64, sem_p 0.38)
- tom was very happy to help his mom  ->  tom was very happy to eat his mom  (syn_p 1.00, sem_p 0.00)
- she saw a boy  ->  she loved a kind  (syn_p 0.00, sem_p 0.00)
- tom and mia are scared  ->  tom and amy are shiny  (syn_p 0.96, sem_p 0.29)
