# Skilled-memory (digit span, SF / DD) protocol and numbers, with the EPAM IV comparison model

Compiled 2026-11-21. Every number below comes from a source that was actually opened and read as text. Where a value could not be read it says "not found".

**Important limitation.** The three primary reports (EC80, CE81, CE82) could NOT be read. All SF numbers here are second-hand (cited by RSS95/ES87), tagged "via". DD's data (a later subject trained with SF's mnemonic) are read from RSS95 and ESC93, which are primary for the EPAM IV model and for Staszewski's DD data as they report them.

## Sources read (and how)

| Tag | Source | Copy read | Caveats |
|---|---|---|---|
| RSS95 | Richman, Staszewski & Simon (1995) "Simulation of expert memory using EPAM IV", Psychological Review 102:305-330 | gwern.net PDF, text extracted (26 PDF pages; journal pages 305-330 are printed in the text and cited below as "p. n") | OCR spacing noise. Figures 1-10 are images (not read); only the text, Tables 1-5 and captions were read. The Appendix (Ur's reliability analysis, from p. 329) was not read. |
| ES87 | Kliegl, Smith, Heckhausen & Baltes (1987) "Mnemonic training for the acquisition of skilled digit memory", Cognition and Instruction 4:203-223 | Potsdam postprint, text extracted (22 PDF pages). Journal page = PDF page + 202. | About other participants (SP, BB). Used only for its descriptions of SF/DD (via) and for a speed-component comparison. Only the abstract, intro, method and two discussion passages (journal pp. 203-209, 216, 222) were read; the SP and BB results sections were not transcribed. |
| ESC93 | Staszewski (1993) "A theory of skilled memory", Proc. 15th CogSci 971-975 | escholarship PDF, text extracted (5 content pages, proceedings pp. 971-975) | OCR noise in some words. |
| EC80 | Ericsson, Chase & Faloon (1980) Science 208:1181-1182 | NOT READ | Science 403; no free text found. Cited only via RSS95, ES87. |
| CE81 | Chase & Ericsson (1981) "Skilled memory", in Anderson (Ed.) pp. 141-189 | NOT READ | archive.org item is borrow-only (no text); no other copy. |
| CE82 | Chase & Ericsson (1982) "Skill and working memory", Psych. Learn. Motiv. 16 | NOT READ | DTIC ADA114634 returns 403 to automated fetch; archive.org item is borrow-only. (RSS95's own reference list prints pp. 141-190 for CE82, apparently a copy error from CE81.) |
| E85, EK95, Staszewski 1988a/1990 | Ericsson (1985); Ericsson & Kintsch (1995); Staszewski's DD papers | NOT READ | No free copies found. DD results below that are credited to Staszewski 1988a/1990 are read only as reported in RSS95/ESC93. |

## Not found / not read

- Everything inside EC80, CE81 and CE82 as primary text: SF's own session-by-session curve, exact SF procedure wording, SF group-size statistics, SF post-session recall percentages, the uncodable-digit-sequence experiment, the letters/consonants transfer span, SF probe/interference/location-cue experiments, the encoding-speed analyses, the three principles in the authors' own words.
- Staszewski 1988a/1990 raw data (only RSS95/ESC93's summaries were read).
- Figures: DD learning curve (RSS95 Fig. 4), pause profile (Fig. 8), retrieval-structure diagrams (Fig. 3, Fig. 7), memory-scanning RTs (Fig. 10).
- RSS95 Appendix (redundancy and reliability analysis; p. 329).
- Any value for SF's span at intermediate points, or for SF's hours beyond the two second-hand figures in Section 3.

---

## 1. Subjects, prior knowledge, practice

**DD (primary-ish; RSS95, ESC93)**
- An undergraduate (ESC93 p. 972), normal adult, who practiced the digit-span task 4-5 days per week under laboratory observation, adapting the mnemonic strategy given him by SF (ESC93 p. 972).
- "More than 3 years of nearly daily training"; "about 865 practice sessions" to reach up to 104 digits (RSS95 p. 306). Elsewhere "850 serial recall sessions, each on a separate day" (RSS95 p. 316; sessions each on a different day, p. 315). ESC93 states "Over 4.5 years, DD's span grew from 7 to 104 digits" (p. 972). These time figures are not reconciled in the sources (3+ years vs 4.5 years).
- Prior knowledge: an experienced runner, "highly familiar with times for standard distances", and "capable of recognizing simple patterns in number sequences" at the outset (RSS95 pp. 306, 313). RSS95 guesses he may have had 500 or more running-time chunks in memory at the start (p. 318, a speculation: "Perhaps he had 500 or more").
- DD developed his retrieval structures "on the basis of advice given to him by SF, the first subject in Chase and Ericsson's pioneering study" (RSS95 p. 315). SF told DD that taking time to search for pattern codes would hurt performance (p. 318).
- Initial span "from the normal range of 7 to 9 digits" (RSS95 p. 316).

**SF and the second runner (via ES87, RSS95; NOT checked against EC80/CE81/CE82)**
- Chase and Ericsson observed "two young adults with otherwise normal memory abilities who acquired digit spans of 82 and 68 in 264 and 286 sessions of laboratory practice" (ES87 p. 204, via CE81/CE82). ES87 does not say which figure belongs to SF and which to DD; do not assign.
- ES87 calls them "two long-distance runners who primarily used their knowledge of running times to code digits" (p. 204), who "generally chunked digits into running times and then created long-term memory retrieval structures by means of chaining and hierarchical ranking. The last digits of a trial were rehearsed in short-term memory" (p. 205).
- Before this work "the best performances on digit-span tasks with fast presentation rates were around 20 digits", displayed by mental calculators or professional mnemonists (ES87 p. 204, via Ericsson 1985).
- RSS95 p. 306: SF used different retrieval-structure patterns from DD "according to Chase and Ericsson's (1982) diagrams"; all are trees with 2 to 4 branches below the root (p. 316).

## 2. Exact digit-span procedure

All from RSS95 pp. 315-316, 319-320 about DD unless stated.
- **Presentation:** experimenter reads random digits aloud "at the rate of one per second" (p. 319; also pp. 306, 316). Digits are not grouped by the experimenter; DD imposes his own grouping (p. 320). DD sits still while they are presented (p. 320).
- **Before the list:** the experimenter announces how many digits; DD says when he is ready; during that interval he prepares a retrieval structure for that length (p. 319).
- **Between presentation and recall:** a rehearsal stage; DD is "quite animated"; reviews the semantic category of each group, moving backward through the groups; tries not to spend long on any group; stops rehearsal on reaching the four 4-digit groups at the beginning of the list (p. 321). Rehearsal and recall times for 100+ digit lists: see Section 7 and Table 5.
- **Recall mode:** spoken. "He repeats those digits back to the experimenter" (p. 319); output generally group by group; if a digit is missing he may go back to the previous group for cues or skip a group and return (p. 322).
- **Staircase:** "Each time that he was able to recall all of the presented digits in order, the next list presented to him was one digit longer, but when he failed on a list the next list was one digit shorter" (p. 316). Consequence stated: about one error for every two lists attempted (p. 316).
- **Span definition:** longest list perfectly recalled on 50% of test trials (Woodworth, 1938) (ESC93 p. 972, footnote 3). ESC93 notes DD has recalled longer lists perfectly on occasion.
- **Trials per session** (RSS95 Table 1, p. 317; "from 3 to 25 lists of random digits", session limited to about 1 hour): sessions 1-15: 25 lists; 16-20: 15; 21-25: 10; 26-40: 8; 41-220: 6; 221-265: 5; 266-475: 4; 476-865: 3.
- **Delay:** no fixed delay stated in RSS95/ESC93. Time to recall for 100+ digit lists is the ordered-recall time in Table 5 (Section 7).
- **SF's own procedure:** not read. (ES87 does not describe it.) The baseline procedure ES87 used for other participants (1.5 s per digit, correct-one-up/wrong-one-down over 210 trials, 50%-success length = span) is ES87's procedure, not SF's (ES87 p. 206).

## 3. Learning curve

- DD: span from 7-9 to a peak of 104 digits over about 850-865 sessions (RSS95 pp. 306, 316). Mean spans are plotted in 5-session blocks (RSS95 Fig. 4; image, not read).
- Text statements from RSS95 p. 318: DD's span tracked EPAM's over the first 500 sessions (first 100 blocks); about "2 years of daily sessions" for the first 100 blocks; then "DD appears stuck at a digit span of about 80" over the next 250 sessions (50 blocks); over the last 100 sessions (20 blocks) he resumed and climbed to about 104. At about block 145 (end of the plateau) his protocols began to report pattern codes more frequently (p. 318). After about 790 sessions his span was about 90 digits (p. 307).
- Temporary ceilings around mean digit spans of 37, 58, 79 and 100 for both DD and EPAM, "points at which both ... have just added three additional four-digit groups to their retrieval structures" (p. 318). RSS95 calls this "mild (and possibly illusory)".
- SF: via ES87 p. 204: spans of 82 and 68 in 264 and 286 sessions (the two runners together; assignment not stated). SF's own milestone values, hours and curve: not found. (Unverified snippets from search results, 7 to about 80 digits, more than 250 hours, 2 years: see Section L.)
- Numbers only from figures: see last section.

## 4. Mnemonic encoding

All DD, RSS95 unless stated.
- Groups of 3 or 4 digits (sometimes 5; may also use a 2-digit chunk and form the 5 as 2+3) coded as meaningful chunks (pp. 306, 310, 312). Retrieval-structure leaves hold 3, 4 or 5 digits (p. 310).
- Categories: DD used 12 semantic interpretation categories (Staszewski 1990); two were split in the model, giving 14 subnets (p. 317): (a) quarter-mile running times, (b) half-mile, (c) three-quarter-mile, (d) 1-mile, (e) 3-km, (f) fast 2-mile (<10 min), (g) slow 2-mile, (h) 3-mile, (i) 10-km, (j) 10-mile, (k) dates, (l) ages (2- and 3-digit, e.g., 29 or 29.2), (m) "double ages" (4-digit, e.g., 8969, treated as pairs of 2-digit ages), (n) miscellaneous salient number patterns (e.g., 222, 468) (p. 317).
- ESC93 p. 972: 12 categories; nine encode digit groups as running times from 1/4 mile to 10 mile; the other categories are ages, dates and numerical patterns. Examples given: "359" = one-mile time, 3 min 59 s, "just under the big [four-minute mile] barrier"; "8201" = 8 min 20.1 s, "a fast college two-mile"; "391" = just under forty years old; "1776" = Revolutionary War started; "987" = sequence descending from nine (ESC93 p. 972).
- Reliability of DD's categorization: simulated at 98% accuracy (ESC93 p. 972); 97% match with an earlier program on 798 groups and 99% agreement with EPAM IV's classifications on the same data (RSS95 p. 325).
- **Share of groups coded:** not found as a single number. Obligatory per ESC93 p. 973: "generation of a semantic code and a retrieval structure address are obligatory operations for each digit group in a list".
- **Pattern codes** (symmetries; Staszewski 1990, 1993): 10 types noticed by the model during study: five in semantic categories (back-to-back categories, identical categories, symmetric categories, ascending progressions, descending progressions), two between groups (back-to-back digits, back-to-back digit pairs), three within a group ("frontwards-backwards", "differences", "add-em-ups") (RSS95 pp. 323-324). Examples: 3663 "frontwards-backwards", 1331 symmetric or "back to back" (pp. 311, 313). ESC93 p. 973 adds: 752 and 3012 "add 'em ups"; 8939 and 7675 "10-apart" and "1-apart"; "X-tenths apart" for similar remote groups (462, 460).
- ESC93 calls pattern codes "contextual codes" (CCs): optional elaborations, unlike the obligatory semantic code and address. Staszewski (1990) showed CC generation is predictable from list contents and speeds serial recall (ESC93 p. 973).
- Timing of encoding in the model: about 1 s per digit presentation; 3-5 s available per group; parameters in Section 9.
- Digit-to-category knowledge held "as images at the nodes of the EPAM discrimination net" (RSS95 p. 310).

## 5. Retrieval structure

- Treelike, hierarchical; leaves are slots for 3-5 digits; DD said he used the same structure each time for same-length lists, with exact organization adjusted to list length (RSS95 pp. 310, 314).
- Function in three steps (RSS95 p. 314): (1) during presentation it guides parsing of the list into 3- and 4-digit sublists, each sorted through the semantic net; (2) each sublist is stored at a node that provides its "address"; (3) at recall, depth-first traversal reactivates addresses in order, each cueing its slot.
- Staszewski's diagrams (RSS95 Fig. 3, p. 315; image, not read) give structures for 25, 50, 75, 100 digits. The model's rules (p. 316): primitive chunks d3, d4, d5; larger: d6=(d3 d3), d7=(d3 d4), d8=(d4 d4), d9=(d3 d3 d3), d10=(d3 d3 d4), d11=(d3 d3 d5), d12=(d4 d4 d4), d13=(d4 d4 d5), d14=(d4 d4 d3 d3), d16=(d4 d4 d4 d4), d21=(d9 d12), d22=(d9 d13), d23=(d9 d14). Constructed as: initial component d16, then series of d21, remainder from d3/d4. Specific: d25=(d16 d9), d50=(d16 d21 d13), d75=(d16 d21 d21 d17), d100=(d16 d21 d21 d21 d21).
- Branching factor at root: at most 4 for lists under 82 digits; 5 for 83-102; 6 for 103-123 (p. 316). The 4-branch limit coincides with DD's plateau near 80 (p. 318, p. 327): suggested explanation, not shown.
- Rehearsal group at the end: DD "consistently rehearses the last two groups" (p. 321) and does not rehearse the first four 4-digit groups (p. 321). EPAM studied only the last group (p. 321). Last digits in SF's trials were rehearsed in STM (ES87 p. 205, via CE81/82).
- Retrieval structure alone is insufficient: forgetting analysis (p. 319). With forgetting 25% per item, 100 digits (30 groups) without redundancy would be recalled about three times in one trillion; requirement .977 reliability per group for 1 correct list in 2 (p. 319).
- **How it changed over practice:** structure elaborated by adding branches as list length grew (p. 310); initial EPAM net had a structure for 37-digit lists (43 nodes, p. 317). DD's structure for 100 digits was altered in Fig. 7B to fit his pause times (p. 322 and 323). The plateau-with-branching explanation is given at pp. 316 and 318.
- **Evidence from recall pauses (DD):** pause times between digits measured by Staszewski (1988a) on 29 lists of 100+ digits (p. 322). Pauses within a group about 200 ms (time to vocalize a digit); between groups: 200 ms plus 250 ms per node traversed in the structure (p. 323). Correlation of EPAM's predicted with DD's actual pause profile: r = .932 with the structure from DD's reports (Fig. 7A), r = .964 with the revised top-level structure (Fig. 7B) (p. 323). The 250 ms per node was "estimated from DD's performance" (p. 313) and matches Staszewski's earlier one-parameter model (p. 323). First three recall spikes (digit locations 4, 8, 12; the unrehearsed groups) are variable, 950-2,700 ms each in EPAM (p. 323).
- Memory-scanning test (DD; about 790 sessions, span 90; 50-digit lists; probe group shown visually, report group that followed or preceded it): DD's error rate 5.8%; times proportional to nodes traversed at group boundaries; no difference between forward and backward (p. 325-326). EPAM error 1.9%, correlation with DD's median times .70 (p. 326). Actual RT values (Fig. 10): not read.

## 6. Post-session recall (free recall) and recognition

- At the end of each practice session DD was asked to recall all digits from that session; he recalled them "not in the order of presentation, but according to the semantic interpretations" (RSS95 pp. 306-307).
- Staszewski's free-recall data: 27 lists of about 100 digits, three lists per session (a 10th session discarded because DD erred on a serial recall). DD reported categories in a fixed order: running-time categories ascending, then dates, ages, miscellaneous; "94%" of groups recalled clustered within categories (strict scoring) and ordered smallest to largest within category; near 100% with a more generous criterion (p. 326).
- DD missed 8% of the groups in free recall; EPAM missed none (p. 326). Groups appearing twice in a session were recalled twice, in immediate succession (p. 327).
- Table 4 (p. 326), nine sessions, 87 groups presented each: groups output by both DD and EPAM 76-85 (for sessions 858-868: 76, 78, 85, 82, 80, 77, 76, 81, 78); same semantic interpretation 76, 78, 83, 81, 80, 74, 75, 81, 77; Spearman correlation of output order .932, .914, .868, .862, .984, .792, .985, .985, .981 (range given in text .782-.985, p. 327; the table in the text has .792 as the minimum cell, small inconsistency in the source or OCR).
- Free-recall time for a three-list session of 100+ digit lists: DD 504 s (Table 5) vs text 492 s (p. 327); EPAM 388 s (Table 5 and text). RSS95 quotes both 504 and 492 for DD; treat as unresolved.
- Percentage of digits recalled after sessions in SF's case, recognition tests: not found. (Unverified snippet: >90% of 200-300 digits; Section L.)

## 7. Manipulations and their results

**DD, pattern-enriched vs pattern-depleted 50-digit lists (Staszewski 1990, via RSS95 p. 323; eight sessions, six 50-digit trials each; DD's span over 100 then)**

Note: RSS95 text says "six digit span trials of 50 digits each", while Table 3 is for "30-Digit Lists". Both are as printed (p. 323 vs Table 3, p. 324); a source inconsistency. Table 2 (p. 323, mean serial recall scores, s):

| Measure | Enriched DD | Enriched EPAM | Depleted DD | Depleted EPAM |
|---|---|---|---|---|
| Percentage correct | 99.8 | 99.4 | 98.8 | 99.6 |
| Rehearsal time | 29.1 | 39.5 | 48.9 | 41.2 |
| Recall time | 43.3 | 46.8 | 66.7 | 61.4 |
| Total time | 72.4 | 86.3 | 115.6 | 102.6 |

(EPAM = mean of 12 runs.) Depleted = lists lacking the patterns DD codes; enriched = more than the usual pattern opportunities (p. 323).

**Trial order (proactive-inhibition-like; DD, RSS95 p. 324, Table 3):** median total recall time (s) by trial order 1-6. Enriched DD 53.5, 55.5, 55.0, 54.5, 73.5, 77.0; EPAM 57.8, 62.9, 69.1, 69.9, 79.6, 77.9. Depleted DD 67.5, 102.5, 113.0, 102.0, 109.0, 112.0; EPAM 67.1, 71.8, 69.7, 83.3, 92.9, 97.2. Both combined DD 61.0, 61.5, 96.5, 78.0, 77.5, 103.5; EPAM 60.8, 63.3, 69.3, 74.6, 84.7, 88.5. Linear fits: EPAM total time = 6 x trial order + 53 s (r2 = .974); DD = 7 x trial order + 56 s (r2 = .544) (p. 324).

**Rehearsal / recall times for 100+ digit lists (Table 5, p. 326):** DD rehearsal 190 s, ordered recall 200 s, free recall 504 s; EPAM 91, 118, 388. Text p. 327 gives 184/92 and 199/118 for rehearsal and ordered recall, a small inconsistency with the table.

**Error rates:** DD and EPAM err more on 4-digit than on 3-digit groups; EPAM on 30 lists of 100+ digits erred on 3.0% of 4-digit groups vs 1.5% of 3-digit groups (p. 319). DD's higher 4-digit error rate is stated but not quantified.

**Forgetting estimate for pattern codes (p. 324):** 288 codes considered; EPAM reported 218 (forgetting 24.3%); DD reported 220 (possible forgetting 23.6%). Basis of the 25% forgetting parameter (p. 319).

**Not found in the opened sources:** the uncodable (non-running-time) sequence experiment; transfer to letters/consonants with a span value; SF probe, interference, location-cue and encoding-speed experiments; SF group-recall-by-location cues. ES87 pp. 216, 222 and the Cowan snippet (Section L) say only that the skill did not transfer to unpracticed material. ES87 pp. 216, 222 (other participants SP, BB) show the same logic: SP, History-Dates model, recalled about 90 digits at a 1 s rate but only when strings were composed from a small pool of 25-100 triplets; with sequences not parseable into his historical knowledge, memory was only slightly better than pretraining, with little transfer to standard digit span (ES87 p. 216; design on pp. 206-210). BB (Digit-Noun model) recalled 80 random digits at a 5-s rate after 70 sessions (abstract, p. 203). Not SF/DD data, so not a signature effect for SF.

## 8. The three principles of skilled-memory theory (as stated in the opened sources)

None is a direct quote from CE81/CE82 (not read). As described in sources that were read:
- ES87 p. 205 (describing Chase and Ericsson 1981; Ericsson 1985): "(a) a mnemonic system for recoding two, three, or four consecutive digits into meaningful chunks such as running times; (b) an effective and flexible long-term memory retrieval structure within which the recoded digits were organized in a multilevel tree structure; and (c) practice-related improvement in encoding and retrieval speed."
- ESC93 p. 971: "as normal adults develop complex cognitive skills, they acquire knowledge structures and memory management strategies that expand working memory capacity. These are the basic tenets of Skilled Memory Theory"; mechanisms enable storage and retrieval of task-relevant information in LTM (Ericsson & Staszewski 1989).
- RSS95 p. 310: fast encoding rests on slot-filling in a learned retrieval structure rather than adding discrimination-net nodes; "items appear to be stored in long-term memory at the rate of one item per second, or ... one chunk every 3 or 4 s" (p. 314), vs about 8 s per new chunk in rote learning (p. 314).
- ES87 p. 204 (via Bugelski; Simon 1986): at least 4 s, or 8 s per chunk, to fix a chunk in LTM without skill.

## 9. Comparison model: EPAM IV (RSS95)

**What it is.** A program (Common Lisp) in the EPAM family (discrimination net + semantic memory + STM + learned strategies); EPAM IV adds a learned retrieval structure (slotted tree in LTM), a depth-first associative search over the net, node/branch activation, simplified learning (one node at a time, each with a partial image), multiple responses per stimulus, an elaborated STM with a background-learning assumption (RSS95 pp. 305, 310).

**Inputs given as initial conditions** (strategies, not fitted): 109-node initial net: 21 nodes for digit-to-word translation, 43 nodes for the retrieval structure for 37-digit lists, 46 root nodes for the 14 category subnets (p. 317); pattern-code productions (p. 318); retrieval-structure chunk patterns d3, d4, d5, etc. (p. 316); hand-built category net matching DD's 798 groups at 99% (p. 325).

**Parameters** (pp. 308, 313, 315, 319-320, 323):
- Recognition: 10 ms per test node; associative (depth-first) search: 250 ms per node.
- Adding a discrimination-net node takes at least 5 s (p. 308); each LTM learning step at least 1.75 s; building new chunks about 8 s (p. 313); 8-10 s when learning a new node during rehearsal (p. 322).
- Per digit: 100 ms to access a digit in the sensory store, 200 ms to attach to a structure, 100 ms to notice a pattern cue (p. 320). Auditory sensory store retention 3,800 ms (about four digits) (p. 320).
- Slot filling about 300 ms (one-syllable word) (p. 315); vocalization 200 ms per digit (p. 322).
- Forgetting: 25% of information attached to retrieval-structure slots (including pattern-code info) lost when attention shifts away; 20% climbs faster to a span above 100; 30% may not reach 100 in 865 sessions (p. 319). No forgetting in the semantic net.
- Search cutoff: 1 simulated minute per digit group (p. 322).
- Presentation rate 1 digit/s; EPAM got the same number of lists per session as DD (Table 1) and the same staircase rule (p. 316).
- Seen as not free: "modest changes in these parameters have little or no effect" except order-of-magnitude errors (p. 320).

**What it reproduced** (RSS95 conclusion, p. 327, with data above):
1. Learning curve: 7-9 digits to about 104 over 865 sessions; two runs matched DD's mean span in 16 of the 5-session blocks (listed p. 318) and "interlaced" over the first 500 sessions (p. 318). Run 1: span reached about 100, then fell back to about 80; run 2: plateau 80-90 between blocks 80-150, then about 110 (p. 318). Net grew from 109 to about 3,400 chunks at session 865 (p. 318); about one new node per list.
2. Roughly one error every two lists (staircase) (p. 316).
3. Pause profile in serial recall: r = .932 (Fig. 7A structure) and r = .964 (Fig. 7B revised structure) (p. 323).
4. Pattern-enriched vs depleted effect: direction reproduced, size smaller (EPAM 86.3 vs 102.6 s total; DD 72.4 vs 115.6 s; Table 2).
5. Trial-order slowing: slopes 6 vs 7 s per trial (p. 324).
6. Free recall by category, clustered 100% (DD 94%), within-category ascending order, Spearman .782-.985 (Table 4); category prediction 99% (p. 325).
7. Memory scanning: r = .70 with DD's median times (p. 326).
8. Four-digit groups err more than three-digit groups (EPAM 3.0% vs 1.5%) (p. 319).
9. Temporary ceilings near spans 37, 58, 79, 100 (p. 318).
10. 23.6% vs 24.3% pattern-code forgetting, supporting the 25% parameter (p. 324).

**What it did NOT reproduce:** the long plateau at about 80 digits (DD, about 250 sessions); EPAM climbs to 100 or plateaus by chance. EPAM underpredicts rehearsal time (about half of DD's) and ordered-recall time (about 60%) on 100+ digit lists (Table 5, p. 327). EPAM never misses a group in free recall (DD missed 8%). EPAM's enrichment effect is smaller than DD's. Digit-group categories and pattern codes are partly hand-given (p. 318). The conclusion section says SF's original data were not simulated; the model is of DD (pp. 306, 327). Not simulated: letter transfer, uncodable lists (not found anywhere).

**Other models.** ESC93 p. 973-975 gives a production-system model (OPS5) of DD's encoding (68 production rules for contextual codes). Result: of 345 CCs in DD's protocols, the model generated 282 (82%); list parsing and semantic code assignment coincided with DD's over 99%; false positives 202 (56 after discounting redundant CCs) (Table 1, p. 974). By type (matches, misses, false positives, conditional false positives): within-group digit CCs 84/31/49/11; adjacent groups 118/5/38/9; remote groups within lists 48/17/41/10; remote groups across lists 32/10/74/26 (p. 974). Model "overly powerful" because no time limits and no forgetting (p. 975).

---

## Signature effects a model must reproduce (human values, with source page)

1. **Digit span grows from about 7 to 80-104 with practice, with 1 digit/s presentation and an up/down staircase** (DD: 7-9 to a peak of 104, 865 sessions, RSS95 pp. 306, 316; SF/second runner: 82 and 68 digits in 264 and 286 sessions, via ES87 p. 204). The span is not an average of one list length; a staircase yields about one error per two lists (RSS95 p. 316).
2. **Staircase session lengths shrink as lists lengthen**: 25 lists per session early, down to 3 lists per session after session 475 (RSS95 Table 1, p. 317).
3. **Plateau at about 80 digits for DD of about 250 sessions** before resuming to about 104 (RSS95 p. 318); coincides with a fifth top-level branch (>82 digits, p. 316). EPAM IV does not produce it.
4. **Temporary ceilings near mean spans 37, 58, 79, 100** (three more 4-digit groups added) (RSS95 p. 318).
5. **Digits recoded in 3-4 digit groups as running times (9 categories from 1/4 to 10 mile), plus ages, dates and number patterns**; 12 categories (ESC93 p. 972); 14 subnets in the model (RSS95 p. 317). Categorization 97-99% reproducible (RSS95 p. 325; ESC93 p. 972).
6. **Hierarchical retrieval structure with 2-4 branches per node (to 82 digits), 3/4/5-digit leaves** (RSS95 p. 316); fixed per list length; guides parsing, addressing, retrieval (p. 314).
7. **Recall pauses**: about 200 ms within a group; about 200 ms + 250 ms per node traversed between groups; pause profile fit r = .932 / .964 (RSS95 p. 323). The first three groups (not rehearsed) have long variable pauses (950-2,700 ms in EPAM) (p. 323).
8. **Rehearsal between presentation and recall**: backward through groups; last two groups always rehearsed; first four 4-digit groups not (RSS95 p. 321). DD rehearsal time on 100+ digit lists about 184-190 s, ordered recall about 199-200 s (RSS95 p. 326-327).
9. **Free recall after the session is organized by semantic category, not presentation order**: 94% of groups clustered within categories, running times ascending first, then dates, ages, miscellaneous; 8% of groups missed (RSS95 p. 326).
10. **Pattern encodings speed recall**: enriched 72.4 s total (29.1 rehearsal, 43.3 recall) vs depleted 115.6 s (48.9, 66.7); accuracy 99.8% vs 98.8% (RSS95 Table 2, p. 323).
11. **Slowing across trials in a session**: DD total time about 7 s per additional list (56 s at trial 0), EPAM 6 s (RSS95 p. 324).
12. **Forgetting from the retrieval structure about 24-25%**; pattern-code forgetting 23.6% for DD (RSS95 pp. 319, 324); redundancy (retrieval structure + semantic code + pattern code) required for error-free 100-digit recall (p. 319).
13. **4-digit groups err more than 3-digit groups** (EPAM 3.0% vs 1.5%; DD "higher" but not quantified) (RSS95 p. 319).
14. **Probe task (after/before a probe group in a 50-digit list)**: DD error 5.8%; times proportional to nodes traversed; no forward/backward difference (RSS95 pp. 325-326).
15. **Skill is material-specific**: it does not carry over to material not parseable into the expert's knowledge (SP/BB analogs, ES87 pp. 216, 222; SF's letter span value not found).

## Numbers available only from figures (not read)

- DD's learning curve: mean span per 5-session block over 175 blocks (RSS95 Fig. 4A, B, p. 317); the only text points are those in Section 3.
- Growth of the EPAM net (Fig. 5, p. 318; text gives 109 to about 3,400 chunks).
- Retrieval structures for 25, 50, 75, 100 digits (Fig. 3, p. 315; Fig. 7A, B, p. 322): group counts and nesting must be taken from the construction rules in Section 5, not from the diagrams.
- DD's median pause time at each digit location 1-104 (Fig. 8, p. 323); only the means in text (200 ms within group; 250 ms per node).
- Median memory-scanning RTs by boundary crossing (Fig. 10, p. 326; roughly 0.5-4 s on the axis).
- Mean recall time vs trial order (Fig. 9, p. 325; Table 3 gives the medians).
- SF's learning curve and per-session data: not found in any opened source.

## Section L. Unverified leads (NOT read in a source; search-result snippets, not page-cited)

- SF: digit span from 7 to about 80 over about 2 years, more than 250 hours of lab practice (snippet quoting a DTIC technical report); the Science 1980 abstract as summarised by search: 7 to 79 after more than 230 hours; after 1 year 70 digits (snippet). Not verified. The sessions figure that was read (264 and 286 sessions, ES87 p. 204) is consistent with an hour per session.
- SF coded 3-4 digit groups as running times, ages and dates added later (snippet).
- Post-session recall over 90% of 200-300 digits (snippet).
- A secondary page, not a primary source, says SF's retrieval structure handled 22 chunked patterns and did not transfer to letters, letter span staying near 7 (Cowan 2013 snippet, no page).
- Presentation about 1 digit per second, +1/-1 staircase, about an hour a day, 2-5 days per week (snippet; the 1 digit/s and staircase facts are confirmed for DD in RSS95 p. 316, but for SF only by snippet).

## Open items to finish (sources still needed)

Primary texts of CE81, CE82 (DTIC ADA114634 403s to automated fetch), EC80; Ericsson (1985); Staszewski (1988a, 1990); Ericsson & Staszewski (1989). With those: SF's hours, span milestones, uncodable-digit and letter-transfer values, post-session recall and group-recall experiments, and the three principles verbatim.
