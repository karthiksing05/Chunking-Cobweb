468 R-sem source sentences (input; R-sem = 1 by construction).

| variant | n | changed words/sent | share unchanged | R-sem | R-syn | R-non | R-syn (changed only) | R-sem (changed only) | R-non (changed only) |
|---|---|---|---|---|---|---|---|---|---|
| pos | 1404 | 1.72 | 0.15 | 0.313 | 0.219 | 0.467 | 0.257 | 0.197 | 0.547 |
| fine | 1404 | 1.68 | 0.15 | 0.368 | 0.255 | 0.377 | 0.299 | 0.258 | 0.443 |
| morph | 1404 | 1.56 | 0.17 | 0.480 | 0.309 | 0.211 | 0.374 | 0.371 | 0.255 |
| morph_ar | 1404 | 1.56 | 0.17 | 0.501 | 0.291 | 0.208 | 0.351 | 0.397 | 0.252 |
| morph1 | 1404 | 0.76 | 0.25 | 0.657 | 0.234 | 0.109 | 0.310 | 0.545 | 0.145 |
| chunk | 1404 | 1.68 | 0.12 | 0.707 | 0.249 | 0.043 | 0.283 | 0.667 | 0.049 |
| chunk1 | 1404 | 1.01 | 0.19 | 0.816 | 0.170 | 0.014 | 0.209 | 0.773 | 0.018 |
| fine_c | 1404 | 1.27 | 0.20 | 0.387 | 0.280 | 0.333 | 0.349 | 0.236 | 0.415 |
| morph_c | 1404 | 1.21 | 0.22 | 0.521 | 0.308 | 0.170 | 0.398 | 0.383 | 0.219 |
| morph_c1 | 1404 | 0.75 | 0.25 | 0.630 | 0.263 | 0.107 | 0.351 | 0.506 | 0.143 |
| chunk_c | 1404 | 0.89 | 0.47 | 0.801 | 0.170 | 0.030 | 0.320 | 0.624 | 0.056 |
| chunk_c1 | 1404 | 0.79 | 0.47 | 0.811 | 0.157 | 0.032 | 0.296 | 0.644 | 0.060 |

### Samples (changed sentences only)

**pos**
- sue and the bird played together all day  ->  sue and the time played together all day  (syn_p 1.00, sem_p 0.04)
- tom and ben are best friends  ->  tom and sara are much friends  (syn_p 0.00, sem_p 0.11)
- he did not get it  ->  he be not gave it  (syn_p 0.00, sem_p 0.00)
- she was very pretty  ->  she was very long  (syn_p 0.02, sem_p 0.00)
- he was scared but wanted to help  ->  he was many but wanted to went  (syn_p 0.00, sem_p 0.00)
- she liked to play with her friend tom  ->  she liked to play with her water tom  (syn_p 0.95, sem_p 0.00)

**fine**
- they liked to play in the water  ->  they liked to play in the room  (syn_p 1.00, sem_p 1.00)
- sue and the bird played together all day  ->  sue and the spot played together all day  (syn_p 1.00, sem_p 0.99)
- she loved to be friends with the bear  ->  she opened to make friends with the fish  (syn_p 0.00, sem_p 0.00)
- they want to run fast  ->  they like to be ever  (syn_p 0.00, sem_p 0.00)
- he did not get it  ->  he made not know it  (syn_p 0.00, sem_p 0.00)
- i was happy he could help  ->  i was happy he will make  (syn_p 0.00, sem_p 0.01)

**morph**
- he liked to play with her toy car  ->  he liked to play with her sky toy  (syn_p 1.00, sem_p 0.27)
- they loved to play near the park  ->  they started to play from the dog  (syn_p 0.00, sem_p 0.00)
- the car was so happy  ->  the room was so happy  (syn_p 0.98, sem_p 0.95)
- it is shiny and red  ->  it is proud and bad  (syn_p 0.36, sem_p 0.00)
- it was so happy to find it  ->  it was so happy to help it  (syn_p 0.17, sem_p 0.04)
- i was a pretty toy bird  ->  i was a red sky girl  (syn_p 0.96, sem_p 0.89)

**morph_ar**
- he did not get out  ->  he walked not get up  (syn_p 0.00, sem_p 0.00)
- she had a big idea  ->  she had a big cat  (syn_p 1.00, sem_p 1.00)
- sue and the bird became good friends  ->  sue and the boy became red friends  (syn_p 0.97, sem_p 0.11)
- the mom said yes let's find it together  ->  the mom said yes things thank it together  (syn_p 0.00, sem_p 0.00)
- she was so happy to find it  ->  she was so happy to see it  (syn_p 1.00, sem_p 1.00)
- tom was very happy to help his mom  ->  tom was very happy to take his mom  (syn_p 0.72, sem_p 0.92)

**morph1**
- he wanted to help the cat  ->  he wanted to find the cat  (syn_p 1.00, sem_p 1.00)
- his dad said be careful tim  ->  his toy said be careful tim  (syn_p 0.02, sem_p 0.13)
- tom loved to play all day  ->  tom decided to play all day  (syn_p 1.00, sem_p 1.00)
- the boy said yes  ->  the room said yes  (syn_p 1.00, sem_p 0.25)
- they decided to run away  ->  they decided to look away  (syn_p 1.00, sem_p 0.99)
- they decided to run away  ->  they started to run away  (syn_p 1.00, sem_p 1.00)

**chunk**
- she wanted to play with her friends  ->  it wanted to play with his friends  (syn_p 1.00, sem_p 1.00)
- but sue was not sad  ->  but something was not sad  (syn_p 0.97, sem_p 0.90)
- he had a friend named bob  ->  it had a car named sam  (syn_p 1.00, sem_p 0.45)
- anna and sam look at each other  ->  tim and ben look at each other  (syn_p 1.00, sem_p 1.00)
- he was so happy he said yes  ->  it was so happy she said yes  (syn_p 0.58, sem_p 0.96)
- she played with the ball all day  ->  it played with the bird all day  (syn_p 1.00, sem_p 1.00)

**chunk1**
- tom loved to play outside in the sun  ->  tom loved to play outside in the tree  (syn_p 1.00, sem_p 1.00)
- the friends were very happy  ->  the toys were very happy  (syn_p 1.00, sem_p 1.00)
- anna was sad  ->  tim was sad  (syn_p 1.00, sem_p 1.00)
- tom was sad  ->  mia was sad  (syn_p 1.00, sem_p 1.00)
- he was happy to have new friends  ->  he was happy to have good friends  (syn_p 1.00, sem_p 1.00)
- he gave it water every day  ->  he gave it food every day  (syn_p 1.00, sem_p 1.00)

**fine_c**
- the cat was happy and said thank you  ->  the tree was happy and said like you  (syn_p 0.00, sem_p 0.00)
- the cat was happy with the toy car  ->  the park was happy with the box cat  (syn_p 0.69, sem_p 0.00)
- they liked to play outside in the sun  ->  they liked to play suddenly in the girl  (syn_p 0.00, sem_p 0.00)
- he was a big red ball  ->  he was a big excited ball  (syn_p 0.02, sem_p 0.02)
- he felt so happy  ->  he looked so happy  (syn_p 1.00, sem_p 1.00)
- he wanted to help the cat  ->  he wanted to look the time  (syn_p 0.00, sem_p 0.01)

**morph_c**
- they were not scared anymore  ->  they were not shiny around  (syn_p 0.01, sem_p 0.14)
- the mom said yes let's find it together  ->  the mom said yes tim's thank it together  (syn_p 0.00, sem_p 0.00)
- the cat was happy too  ->  the room was happy too  (syn_p 0.97, sem_p 0.99)
- lily and lucy felt sad  ->  lily and lucy got sad  (syn_p 1.00, sem_p 1.00)
- the boy was happy  ->  the park was happy  (syn_p 1.00, sem_p 0.98)
- i was a pretty toy bird  ->  i was an unexpected kind food  (syn_p 0.08, sem_p 0.00)

**morph_c1**
- they loved to play in the sun  ->  they loved to play in the park  (syn_p 1.00, sem_p 1.00)
- they do not know him  ->  they do not see him  (syn_p 1.00, sem_p 0.91)
- they were so happy and excited to go  ->  they were so happy and surprised to go  (syn_p 0.13, sem_p 0.75)
- everyone was so happy and excited  ->  everyone was so happy and careful  (syn_p 0.50, sem_p 0.87)
- ben and ben smiled  ->  ben and ben thought  (syn_p 0.25, sem_p 0.13)
- one day she had a big door  ->  one day she had a big lot  (syn_p 0.47, sem_p 0.05)

**chunk_c**
- they wanted to play in the park  ->  they wanted to play in the ball  (syn_p 0.31, sem_p 0.20)
- they wanted to play at the park  ->  they wanted to play at the water  (syn_p 1.00, sem_p 1.00)
- the little bird thought about it  ->  the little girl thought about it  (syn_p 1.00, sem_p 1.00)
- he played and played with the ball  ->  he played and played with the water  (syn_p 1.00, sem_p 1.00)
- one day max saw a little bird  ->  one day max saw a big room  (syn_p 1.00, sem_p 1.00)
- amy was so happy and excited  ->  something was so happy and excited  (syn_p 0.92, sem_p 0.93)

**chunk_c1**
- the cat was playing with a ball  ->  the dog was playing with a ball  (syn_p 1.00, sem_p 1.00)
- the boy said the little girl went outside  ->  the girl said the little girl went outside  (syn_p 1.00, sem_p 0.69)
- his mom smiled and said yes tim  ->  her mom smiled and said yes tim  (syn_p 0.11, sem_p 0.89)
- his mom smiled and said yes tim  ->  her mom smiled and said yes tim  (syn_p 0.11, sem_p 0.89)
- his mom looked at the box  ->  his mom looked at the water  (syn_p 1.00, sem_p 1.00)
- he liked to play with his ball  ->  he liked to play with your dad  (syn_p 1.00, sem_p 0.84)
