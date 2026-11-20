"""Round 221b: parse then regenerate, with only existing parts. Held-out real positions (shuffled training lines [10000:10300], never
learned and disjoint from the judge's real lines 19459-27459) are PARSED by the existing parser (GraphTrellis.parse, learning off);
each parsed chunk is REPLACED by a stored instance (pieces at offsets) drawn from the content concept kc levels above the leaf its
content sorts to (L.cnt.categorize on its content; the concept's stored top chunks, as the code's copy route), placed at the chunk's
own anchor; when no draw fits there (30 tries; squares free or the board edge), the chunk stays. kc in KCS (99 = root).
Variants: "keep" = the leftover single pieces (kings included) stay; "resample" = each single is redrawn: kc=0 its own token (unchanged);
0<kc<root a token from the stored single pieces on that square; root a token from all stored single pieces (the top alphabet's single
piece-square items). Identity (no replacement) is the real-board control. Same judge (r220), same learner as r220/r221.
Usage (compute node, repo root): r221_parse_regen.py <seed> <N list> <replicates> <tag>
"""
import itertools, json, os, random, sys, time, bisect
from collections import Counter

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402
import r220_judge as J  # noqa: E402

seed = int(sys.argv[1]); NS = [int(x) for x in sys.argv[2].split(",")]; REP = int(sys.argv[3]); tag = sys.argv[4]
KCS = [0, 1, 2, 3, 4, 6, 99]
CFG = json.loads('{"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,'
                 '"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"}')
OUTF = os.path.join(ROOT, "experiments", "v3", "results", "chess_regimes", f"r221_pr_{tag}_s{seed}.json")
judge, rng = J.Judge(), random.Random(seed)
lines = J.LINES[:19459]
random.Random(seed).shuffle(lines)
held = lines[10000:10300]
L = GraphTrellis(max_size=4, seed=seed, **CFG)
singles, chunks = Counter(), []


def learn_one(f):
    pos = C.parse_fen(f)
    r = L.learn(pos)
    sqs, toks = r["exp"].data[1], r["exp"].tokens
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            singles[(e.token, sqs[i])] += 1
        elif e.rid is not None:
            a = sqs[min(e.span)]
            chunks.append((e.rid, frozenset((toks[i], sqs[i][0] - a[0], sqs[i][1] - a[1]) for i in e.span)))


def wchoice(items, cum):
    return items[bisect.bisect(cum, rng.random() * cum[-1])]


def run(name, boards, N, res):
    c, ex, causes, fallback, tot = Counter(), {"real": [], "legal": [], "non": []}, Counter(), 0, 0
    for b, fb, n in boards:
        r = judge.classify(b)
        c[r] += 1
        fallback += fb; tot += n
        if r == "non":
            st = chess.Board(J.to_fen(b)).status()
            for fl in chess.Status:
                if fl and st & fl and fl != chess.STATUS_VALID:
                    causes[fl.name] += 1
        if len(ex[r]) < 4:
            ex[r].append(J.to_fen(b).split()[0])
    n = len(boards)
    row = {k: round(c[k] / n, 3) for k in ("real", "legal", "non")}
    row |= {"chunks kept (no draw fit)": round(fallback / max(tot, 1), 3), "illegal_causes": {k: round(v / n, 3) for k, v in causes.items()}, "examples": ex}
    res[f"N{N}"][name] = row
    print(json.dumps({"N": N, "arm": name, **{k: v for k, v in row.items() if k not in ("examples", "illegal_causes")}}), flush=True)


res = {"seed": seed, "NS": NS, "n": len(held) * REP, "tau": judge.tau, "kcs": KCS}
done, t0 = 0, time.time()
for N in NS:
    for f in lines[done:N]:
        learn_one(f)
    done = N
    L._quiet = True
    paths = {}
    for rid in {c[0] for c in chunks}:
        try:
            paths[rid] = L.cnt.categorize(L.records[rid].cnt)
        except Exception:
            pass
    under, tok_sq, tok_all = {}, {}, Counter()
    for rid, rel in chunks:
        for node in paths.get(rid, []):
            under.setdefault(node.id, Counter())[rel] += 1
    for (t, sq), n in singles.items():
        tok_sq.setdefault(sq, Counter())[t] += n
        tok_all[t] += n
    cache = {}

    def draw(counter):
        k = id(counter)
        if k not in cache:
            its = list(counter)
            cache[k] = (its, list(itertools.accumulate(counter[i] for i in its)))
        return wchoice(*cache[k])

    parsed = []                                   # per held-out position: its board, units
    for f in held:
        pos = C.parse_fen(f)
        r = L.parse(pos)
        sqs, toks = r["exp"].data[1], r["exp"].tokens
        units = []
        for e in r["top"]:
            pcs = {sqs[i]: toks[i] for i in e.span}
            if e.token is None and len(e.span) > 1:
                path = L.cnt.categorize(L._content(e))
                units.append(("c", pcs, sqs[min(e.span)], [n.id for n in path]))
            else:
                units.append(("s", pcs, None, None))
        parsed.append((pos, units))
    res[f"N{N}"] = {"learned": N, "learn_s": round(time.time() - t0), "chunk units": len(chunks),
                    "parsed chunks per position": round(sum(sum(u[0] == "c" for u in us) for _, us in parsed) / len(parsed), 2)}
    print(json.dumps({"N": N, "learned": N, "s": round(time.time() - t0), **{k: v for k, v in res[f"N{N}"].items() if k.startswith("parsed")}}), flush=True)
    run("identity (real board, parse units unchanged)", [(pos, 0, 0) for pos, _ in parsed for _ in range(REP)], N, res)
    for variant in ("keep", "resample"):
        for kc in KCS:
            out = []
            for pos, units in parsed:
                for _ in range(REP):
                    board = {}
                    for kind, pcs, a, path in units:
                        board.update(pcs)
                    order = [u for u in units if u[0] == "c"]
                    rng.shuffle(order)
                    fb = 0
                    for kind, pcs, a, path in order:
                        for s in pcs:
                            del board[s]
                        j = max(0, len(path) - 1 - kc)
                        while path[j] not in under and j > 0:        # a concept no stored top chunk sits under: its nearest ancestor that has some
                            j -= 1
                        node = path[j]
                        got = None
                        for _ in range(30):
                            rel = draw(under[node])
                            new = {(a[0] + dx, a[1] + dy): t for t, dx, dy in rel}
                            if all(0 <= x < 8 and 0 <= y < 8 and (x, y) not in board for x, y in new):
                                got = new
                                break
                        if got is None:
                            got, fb = pcs, fb + 1
                        board.update(got)
                    if variant == "resample" and kc > 0:
                        for kind, pcs, a, path in units:
                            if kind == "s":
                                ((sq, t),) = pcs.items()
                                src = tok_all if kc == 99 or sq not in tok_sq else tok_sq[sq]
                                board[sq] = draw(src)
                    out.append((board, fb, len(order)))
            run(f"{variant} kc={kc}", out, N, res)
    json.dump(res, open(OUTF, "w"), indent=1)
print("done", round(time.time() - t0), flush=True)
