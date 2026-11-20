"""Build update_presentation batches that add the figure slides to the TRELLIS v2 Google Slides deck.
Usage: python3 gen3.py   -> writes fig_batch1.json (slides, text, images, renumber) and fig_notes.json (notes text by slide id)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
U = "https://drive.google.com/uc?export=download&id="
GREY = {"red": 0.4196, "green": 0.4471, "blue": 0.5020}
DARK2 = {"themeColor": "DARK2"}
# (id, after_old_slide, title, kind, image file id, (w,h px), layout params, lead, takeaway, source, notes)
D = [
 dict(id="f01", after=12, title="Generation Across Experience", img="1ypJduoWSORNEx6N-DERulH9FYryMgi7n", px=(1440,920), kind="S", maxw=4900000, maxh=3130000,
  lead="New, grammatical, novel sentences per attempted draw, mean of 8 seeds.",
  take="TRELLIS matches the LSTM at 250, 500 and 1,000 sentences and beats GPT at 250 and 500.",
  src="Source: r212_seeds.md (Qwen2.5-3B judge).",
  notes="This is the generation result along the learning curve. For each training size we count the sentences per attempted draw that are distinct, new, judged grammatical and contain a trigram the training set never stored, and we average over eight seeds. At 250 sentences TRELLIS scores .242 against .232 for the LSTM and .207 for GPT; at 500 sentences .298 against .299 and .274; at 1,000 sentences .328 against .335 and .328. The paired differences against the LSTM include zero at every size. The advantage over GPT holds at 250 sentences (+.035) and 500 sentences (+.024), with confidence intervals above zero, and disappears at 1,000 sentences (-.0002). So with little data TRELLIS writes as well as a matched LSTM and better than a matched GPT, and with more data it only ties."),
 dict(id="f02", after=16, title="Parsing Along Training", img="1lELtUs1v4ug0LYcOlbkddFSntl0t1pDV", px=(2200,919), kind="W", maxw=7000000, maxh=2450000,
  lead="Broken sentences accepted as training grows, mean of 4 seeds; lower is better.",
  take="The parse alone barely separates broken sentences; the generator helps most on scrambles.",
  src="Real sentences are accepted .90 to .94 by every score. Source: r216_figure.tsv.",
  notes="This figure draws the table on the previous slide as curves, for every training size from 125 to 4,000 sentences. The threshold of each score is fixed on real sentences, so the dotted line marks the share of real sentences accepted. On scrambled sentences the parse alone falls only from .88 to .77, while the generator falls from .39 to .05, which is close to the LSTM and GPT at .01 for 4,000 sentences. On content swaps every score falls more slowly: the generator goes from .86 to .58, and the LSTM and GPT from .80 to .41 and .40. The curves show that the generator needs little data to catch scrambles but stays behind the neural models on meaning."),
 dict(id="f03", after=18, title="Story Chunks Through Experience", img="1FwgXaVrZaMPvbmW1TcY6_A5gpFqdDgVn", px=(2459,918), kind="W", maxw=7800000, maxh=2450000,
  lead="The same six held-out sentences after 250 to 4,000 training sentences.",
  take="Some chunks, like tim had and an idea, appear at 250 sentences; toy car appears only at 4,000.",
  src="Square brackets: recognised chunks. Round brackets: chosen chunks below the threshold.",
  notes="Each row is one held-out sentence and each column a model trained on more sentences. Square brackets mark chunks whose span posterior clears the recognition threshold, and round brackets mark chunks the parser chose but did not recognise. The sentence tim had an idea is bracketed as tim had and an idea from 250 sentences on. The chunk i loved is recognised at 500 sentences and again at 4,000, and toy car in i loved that toy car appears only at 4,000, when the sentence is bracketed as i loved and that toy car. Grey text under each sentence gives the phrase labels of the two most confident recognised chunks. Chunks arrive one at a time as experience accumulates, which is the incremental learning the theory predicts."),
 dict(id="f04", after=18, title="Phrase Chunks Over Experience", img="1aHXxmAXSVuBwXhhWz7DEILbuHof1dxon", px=(1530,1125), kind="S", maxw=4800000, maxh=3560000,
  lead="Share of committed chunks with each phrase label, 250 to 4,000 sentences.",
  take="Chunks that cross a phrase boundary fall from .29 to .20, against .34 for random spans.",
  src="Dashed: random-span chance; dotted: same-tag chance; green: held-out recognised.",
  notes="Each panel follows one phrase label across training sizes, with the committed chunks in blue, random spans and same-tag spans as chance lines, and held-out recognised chunks in green. The share of chunks that cross a phrase boundary falls from .29 to .20 as training grows, against about .34 for random spans, so the learned chunks respect phrases more and more even though phrases were never labelled. Noun phrases rise from .069 to .082 and verb phrases rise as well, both above their chance lines. The whole sentence stays near .23 throughout. We read this as chunks drifting toward linguistic phrases without supervision, while remaining far from a full match."),
 dict(id="f05", after=20, title="Frequent Chess Chunks", img="15Qh-7QQadv2GJtOrnuLsAEMxeUEp1bJF", px=(1563,850), kind="S", maxw=5300000, maxh=2900000,
  lead="The most frequent chunks in middle-game positions, each drawn at its most frequent place.",
  take="The common chunks pair a king with a rook, or a pawn with a bishop.",
  src="Counts are occurrences in the training positions.",
  notes="These are real chunks from a TRELLIS model trained on middle-game positions, each drawn on the squares of its most frequent instance. The most frequent are black pawn and bishop on g6 and g7, seen 2,164 times, black rook and king on f8 and g8, seen 2,110 times, and white bishop and pawn on g2 and g3, seen 1,826 times. The king and rook pairs appear in several shapes, such as rook and king two files apart, seen 536 times, and king and rook on e8 and h8, seen 380 times, which is how a castled king looks in different stages of the game. The labels in brackets give the piece types and the offset between them."),
 dict(id="f06", after=20, title="Chess Chunks Through Experience", img="1s0EjeV7YVTL-OfjRz3sjFG_25-pynL46", px=(1800,1740), kind="S", maxw=3700000, maxh=3560000,
  lead="Four held-out positions after 100, 300, 1,000, 2,000 and 4,000 learned positions. Recognised chunks are ringed.",
  take="White's castled king is recognised first, at 300 positions; pawn chains come later.",
  src="Each colour is one recognised chunk; the ideas are listed under each board in the source figure.",
  notes="Each row is a held-out position and each column a model with more positions learned. Nothing is recognised at 100 positions. By 300 positions the first chunk to appear is white's castled king, then back-rank groups follow, and by 2,000 to 4,000 positions the model also recognises pawn chains, phalanxes and batteries on both sides. The ideas listed under each board in the source figure come from a separate labelling that the model never saw. We use these boards to show that the chunks are chess structures, and that they appear in an order, simple shelters first and larger pawn structures later."),
 dict(id="f07", after=20, title="One Position, Real and Scrambled", img="1mY2nflIK1cyyr7FfaYl4Usm-fyYxTGwN", px=(728,1131), kind="S", maxw=2400000, maxh=3640000,
  lead="One held-out position and its scrambled twin after 0, 1,000 and 4,000 learned positions.",
  take="Recall of the real position rises from .25 to .86; the scrambled twin reaches .57.",
  src="7 units kept; recalled pieces are drawn in full, each chunk ringed in its own colour.",
  notes="This is the Chase and Simon contrast on a single position. With no learning, the model keeps 7 of 28 pieces, a recall of .25, for both the real position and its scrambled twin. After 1,000 learned positions the real position is recalled at .82, with 23 of 28 pieces held in 6 chunks, while the scrambled twin stays at .25 with no chunks. After 4,000 positions the real position reaches .86 with 24 pieces, and the scrambled twin reaches .57 with 16 pieces in 4 chunks. Experience helps real positions most, and it helps scrambled ones only through the chunks that happen to occur in them."),
 dict(id="f08", after=21, title="Compression Across Experience", img="1SiAmq2iLw1jeXTaVS_psnnXCIiIRQIQ4", px=(1903,605), kind="W", maxw=7800000, maxh=2450000,
  lead="Bits to code a held-out chess position against positions learned, on four test sets.",
  take="On boards shifted one file, TRELLIS needs 147 bits against 150 for BPE on shapes and 206 on squares.",
  src="Control: BPE on squares 109 bits, TRELLIS 128 at 4,000 positions.",
  notes="This is the compression learning curve behind the previous slide. Each panel gives the bits needed to code a held-out position as the model learns from 300 to 4,000 positions. On the control set BPE on squares reaches 108.8 bits and BPE on shapes 111.3, ahead of xz at 122.7 and TRELLIS at 127.7. On other openings TRELLIS reaches 138.3 against 121.4 for BPE on shapes, and on the other wing 154.6 against 141.2. On boards shifted one file, which the learned squares never saw, TRELLIS needs 147.2 bits, BPE on shapes 149.7, xz 181.3, Krimp 198.3 and BPE on squares 205.5, so chunks defined by piece type and offset transfer where chunks tied to squares do not. We report the control honestly: on familiar boards simple compressors are ahead."),
 dict(id="f09", after=22, title="Recall Across Experience", img="1_wL25xkDKtVj3B8UTs68k4GyBLRH9SCp", px=(2400,600), kind="W", maxw=8200000, maxh=2200000,
  lead="Trial-1 recall against positions learned; squares mark people (B, A, M).",
  take="Middle-game recall rises from 7.2 to 12.9 pieces, and passes the master's 16 at 16,000 positions.",
  src="Blue: 7 chunk slots; red: 4 slots; dashed: long stream to 27,000 positions.",
  notes="These four panels are the learning curves of the Chase and Simon recall task. On middle games, trial-1 recall with 7 slots rises from 7.2 pieces at 300 positions to 9.7 at 1,000 and 12.9 at 4,000; with 4 slots it rises from 4.3 to 6.7 and 9.5. People recall 4 pieces as beginners, 8 as class A players and 16 as masters. On the long stream, recall with 7 slots reaches 15.6 pieces at 8,000 positions and 16.7 at 16,000. End games climb from 5.8 to 7.4 pieces against human 4, 7 and 8. On random positions the model also gains with experience (4.2 to 6.0 pieces with 7 slots), whereas the human master falls to 2, and we report that mismatch. The right panel shows that the first chunk the model uses holds 3.9 pieces at 1,000 and at 4,000 positions, against 3.8 for the human master."),
 dict(id="f10", after=22, title="Chase and Simon Trials", img="1sJ0FEIsQJ9WKyq5nIBGFGcykBt0uTVSh", px=(1587,1077), kind="S", maxw=5000000, maxh=3400000,
  lead="Pieces placed correctly over repeated trials: people against TRELLIS memories of three sizes.",
  take="On middle games TRELLIS keeps the human ordering of master, class A and beginner.",
  src="M master, A class A, B beginner; TRELLIS memory after 4,000, 1,000 and 300 positions; previous reconstruction kept.",
  notes="This is the original comparison to Chase and Simon, trial by trial. People on middle games place 16, 8.5 and 4.5 pieces on trial 1 as master, class A and beginner. TRELLIS memories after 4,000, 1,000 and 300 positions place 19.1, 17.3 and 14.5, keep the same ordering, and reach the full board by trial 3. This variant keeps the previous reconstruction between trials, so its trial-1 numbers are higher than the memory-limited curves on the previous slide. On random positions people place only 2 to 3 pieces on trial 1, whereas TRELLIS places more. We read the ordering as agreeing with the expertise effect and the level as showing where the model differs from people."),
 dict(id="f11", after=22, title="Chess Ideas Over Experience", img="1cZKyC3j6jIewVp8StWBSipF8PW5EV8zB", px=(1920,1125), kind="S", maxw=5000000, maxh=2930000,
  lead="Share of committed chunks carrying each chess idea against positions learned.",
  take="Pawn chains rise from .19 to .29 against .16 by chance; pawn rams and attacks stay below chance.",
  src="Dashed: nearby-piece chance; thin lines: seeds.",
  notes="Each panel follows one chess idea, such as a pawn chain, a phalanx or a battery, as the share of committed chunks that carry it, with the share expected for nearby pieces as a dashed chance line and one thin line per seed. The share of chunks carrying a pawn chain rises from about .19 to .29 against about .16 by chance, and pawn phalanxes rise from about .20 to .30 against .16. Batteries and undeveloped pieces stay above chance. Pawn rams and attacks stay far below it. The model never saw these labels, so the rise shows that chunks gather around structures that players name."),
]
# new slide index order
order = []
for old in range(1, 30):
    order.append(("old", old))
    for d in D:
        if d["after"] == old:
            order.append(("new", d["id"]))
newidx = {}
for i, (k, v) in enumerate(order):
    newidx[(k, v)] = i
reqs = []
def box(sid, oid, x, y, w, h, text, runs, align=None):
    r = [{"createShape": {"objectId": oid, "shapeType": "TEXT_BOX", "elementProperties": {"pageObjectId": sid,
          "size": {"width": {"magnitude": w, "unit": "EMU"}, "height": {"magnitude": h, "unit": "EMU"}},
          "transform": {"scaleX": 1, "scaleY": 1, "translateX": x, "translateY": y, "unit": "EMU"}}}},
         {"insertText": {"objectId": oid, "insertionIndex": 0, "text": text}}]
    pos = 0
    for part, st in runs:
        s = {"fontFamily": st["f"], "fontSize": {"magnitude": st["sz"], "unit": "PT"}, "bold": st.get("b", False),
             "foregroundColor": {"opaqueColor": ({"rgbColor": GREY} if st.get("grey") else DARK2)}}
        r.append({"updateTextStyle": {"objectId": oid, "textRange": {"type": "FIXED_RANGE", "startIndex": pos, "endIndex": pos + len(part)},
                  "style": s, "fields": "fontFamily,fontSize,bold,foregroundColor"}})
        pos += len(part)
    if align:
        r.append({"updateParagraphStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": {"alignment": align}, "fields": "alignment"}})
    return r
for d in D:
    sid = d["id"] + "_slide"
    idx = newidx[("new", d["id"])]
    reqs.append({"createSlide": {"objectId": sid, "insertionIndex": idx, "slideLayoutReference": {"layoutId": "p12"}}})
    reqs += box(sid, d["id"] + "_title", 457200, 274320, 8229600, 457200, d["title"] + "\n", [(d["title"] + "\n", dict(f="Merriweather", sz=24, b=True))])
    pw, ph = d["px"]
    ar = pw / ph
    w = min(d["maxw"], d["maxh"] * ar); h = w / ar
    if d["kind"] == "W":
        reqs += box(sid, d["id"] + "_lead", 457200, 868680, 8229600, 320040, d["lead"] + "\n", [(d["lead"] + "\n", dict(f="Lato", sz=14))])
        ix = int(457200 + (8229600 - w) / 2); iy = 1290000
        reqs.append({"createImage": {"objectId": d["id"] + "_img", "url": U + d["img"], "elementProperties": {"pageObjectId": sid,
            "size": {"width": {"magnitude": int(w), "unit": "EMU"}, "height": {"magnitude": int(h), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": 1, "translateX": ix, "translateY": iy, "unit": "EMU"}}}})
        ty = 3790000
        reqs += box(sid, d["id"] + "_take", 457200, ty, 8229600, 520000, d["take"] + "\n", [(d["take"] + "\n", dict(f="Lato", sz=16, b=True))])
        reqs += box(sid, d["id"] + "_ft", 457200, 4343400, 8229600, 320040, d["src"] + "\n", [(d["src"] + "\n", dict(f="Lato", sz=12, grey=True))])
    else:
        y0 = 950000
        reqs.append({"createImage": {"objectId": d["id"] + "_img", "url": U + d["img"], "elementProperties": {"pageObjectId": sid,
            "size": {"width": {"magnitude": int(w), "unit": "EMU"}, "height": {"magnitude": int(h), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": 1, "translateX": 457200, "translateY": y0, "unit": "EMU"}}}})
        tx = int(457200 + w + 320040); tw = 8686800 - tx
        t = d["lead"] + "\n" + d["take"] + "\n" + d["src"] + "\n"
        reqs += box(sid, d["id"] + "_body", tx, y0, tw, 3500000, t, [(d["lead"] + "\n", dict(f="Lato", sz=14)),
                    (d["take"] + "\n", dict(f="Lato", sz=16, b=True)), (d["src"] + "\n", dict(f="Lato", sz=12, grey=True))])
        reqs.append({"updateParagraphStyle": {"objectId": d["id"] + "_body", "textRange": {"type": "ALL"}, "style": {"spaceBelow": {"magnitude": 12, "unit": "PT"}}, "fields": "spaceBelow"}})
    num = newidx[("new", d["id"])] + 1
    reqs += box(sid, d["id"] + "_num", 8229600, 4663440, 457200, 365760, "%d\n" % num, [("%d\n" % num, dict(f="Lato", sz=12, grey=True))], align="END")
# renumber old slides whose position changed
for old in range(1, 30):
    n = newidx[("old", old)] + 1
    if n != old:
        oid = "t%02d_num" % old
        reqs.append({"deleteText": {"objectId": oid, "textRange": {"type": "ALL"}}})
        reqs.append({"insertText": {"objectId": oid, "insertionIndex": 0, "text": str(n)}})
json.dump(reqs, open(os.path.join(HERE, "fig_batch1.json"), "w"))
json.dump({d["id"] + "_slide": d["notes"] for d in D}, open(os.path.join(HERE, "fig_notes.json"), "w"))
print(len(reqs), "requests;", len(json.dumps(reqs)), "bytes;", [(d["id"], newidx[("new", d["id"])] + 1) for d in D])
