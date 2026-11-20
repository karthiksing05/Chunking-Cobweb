"""Round 227: WHERE split into FILE and RANK (r226 where, with one facet of the anchor abstracted at a time) and WHAT, parse then regenerate, only existing reads of the chunk and concept trees.
Held-out real positions (shuffled training lines [10000:10300]) are PARSED by the existing parser; each unit (chunk or single piece) keeps
its place in the parse. Two reads of the concept tree at level kc (levels above the leaf its content sorts to in L.cnt; 99 = root):
  where  the unit's anchor is redrawn from the anchors of the stored top chunks under that concept (the code's anchor read, `anchor_bits`),
         placed where the shape fits and the squares are free (30 tries, else the unit stays). kc=0: the anchors stored under its own leaf.
         kc=99 (root): any square where the shape fits (uniform). A single piece is a 1-piece chunk: kc=0 the stored squares of singles of
         its own token; 0<kc<99 the stored squares of all single pieces; 99 any free square.
  what   the unit's pieces (shape + piece kinds) are replaced by a stored instance under that concept (the copy route, r221b); a single's
         token by a token of the stored singles on its square (kc=0 unchanged; 99 all singles).
Arms: identity (real board), where-only (A, kc), where+what both at kc (B), list controls (counts off / on: pieces iid from piece-square
frequencies; counts on: the stored profile's pieces). Same r220 judge. Usage: r227_file_rank.py <seed> <N list> <replicates> <tag>
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
KINDS = sorted(C.TOKENS)
CFG = json.loads('{"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,'
                 '"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"}')
OUTF = os.path.join(ROOT, "experiments", "v3", "results", "chess_regimes", f"r227_{tag}_s{seed}.json")
judge, rng = J.Judge(), random.Random(seed)
lines = J.LINES[:19459]
random.Random(seed).shuffle(lines)
held = lines[10000:10300]
L = GraphTrellis(max_size=4, seed=seed, **CFG)
singles, chunks, raw_use, sizes, profiles = Counter(), [], Counter(), [], Counter()


def learn_one(f):
    pos = C.parse_fen(f)
    raw_use.update((t, sq) for sq, t in pos.items())
    sizes.append(len(pos))
    cnt = Counter(pos.values())
    profiles[tuple(cnt.get(k, 0) for k in KINDS)] += 1
    r = L.learn(pos)
    sqs, toks = r["exp"].data[1], r["exp"].tokens
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            singles[(e.token, sqs[i])] += 1
        elif e.rid is not None:
            a = sqs[min(e.span)]
            chunks.append((e.rid, a, frozenset((toks[i], sqs[i][0] - a[0], sqs[i][1] - a[1]) for i in e.span)))


def wchoice(items, cum):
    return items[bisect.bisect(cum, rng.random() * cum[-1])]


def adjacent_kings(b):
    ks = [s for s, t in b.items() if t[1] == "K"]
    return any(max(abs(x[0] - y[0]), abs(x[1] - y[1])) == 1 for i, x in enumerate(ks) for y in ks[i + 1:])


def run(name, boards, N, res):
    c, ex, causes, kept, tot, adj, nonn, chk = Counter(), {"real": [], "legal": [], "non": []}, Counter(), 0, 0, 0, 0, 0
    CHECKS = {"OPPOSITE_CHECK", "TOO_MANY_CHECKERS", "IMPOSSIBLE_CHECK"}
    for b, fb, n in boards:
        r = judge.classify(b)
        c[r] += 1
        kept += fb; tot += n
        if r == "non":
            nonn += 1
            adj += adjacent_kings(b)
            st = chess.Board(J.to_fen(b)).status()
            names = {fl.name for fl in chess.Status if fl and st & fl and fl != chess.STATUS_VALID}
            for nm in names:
                causes[nm] += 1
            chk += bool(names) and names <= CHECKS
        if len(ex[r]) < 4:
            ex[r].append(J.to_fen(b).split()[0])
    n = len(boards)
    row = {k: round(c[k] / n, 3) for k in ("real", "legal", "non")}
    row |= {"illegal only by check (share of all boards)": round(chk / n, 3), "units kept (no draw fit)": round(kept / max(tot, 1), 3), "adjacent kings among illegal": round(adj / max(nonn, 1), 3),
            "illegal_causes_of_all_boards": {k: round(v / n, 3) for k, v in causes.items()}, "examples": ex}
    res[f"N{N}"][name] = row
    print(json.dumps({"N": N, "arm": name, **{k: v for k, v in row.items() if k not in ("examples", "illegal_causes_of_all_boards")}}), flush=True)


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
    under, anc_under, tok_sq, tok_all, sq_tok, sq_all = {}, {}, {}, Counter(), {}, Counter()
    for rid, a, rel in chunks:
        for node in paths.get(rid, []):
            under.setdefault(node.id, Counter())[rel] += 1
            anc_under.setdefault(node.id, Counter())[a] += 1
    for (t, sq), n in singles.items():
        tok_sq.setdefault(sq, Counter())[t] += n
        tok_all[t] += n
        sq_tok.setdefault(t, Counter())[sq] += n
        sq_all[sq] += n
    cache = {}

    def draw(counter):
        k = id(counter)
        if k not in cache:
            its = list(counter)
            cache[k] = (its, list(itertools.accumulate(counter[i] for i in its)))
        return wchoice(*cache[k])

    parsed = []
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
                units.append(("s", pcs, sqs[min(e.span)], None))
        parsed.append((pos, units))
    res[f"N{N}"] = {"learned": N, "learn_s": round(time.time() - t0), "chunk units": len(chunks),
                    "parsed chunks per position": round(sum(sum(u[0] == "c" for u in us) for _, us in parsed) / len(parsed), 2)}
    print(json.dumps({"N": N, "learned": N, "s": round(time.time() - t0)}), flush=True)
    run("identity (real board, parse units unchanged)", [(pos, 0, 0) for pos, _ in parsed for _ in range(REP)], N, res)

    def regen(kc, what, facet):
        out = []
        for pos, units in parsed:
            for _ in range(REP):
                board = {}
                for kind, pcs, a, path in units:
                    board.update(pcs)
                order = units[:]
                rng.shuffle(order)
                kept = 0
                for kind, pcs, a, path in order:
                    for s in pcs:
                        del board[s]
                    ok = None
                    if kind == "c":
                        j = max(0, len(path) - 1 - kc)
                        while path[j] not in under and j > 0:
                            j -= 1
                        node = path[j]
                        own = [(t, s[0] - a[0], s[1] - a[1]) for s, t in pcs.items()]
                        for _ in range(30):
                            rel = draw(under[node]) if what else own
                            b0 = (rng.randrange(8), rng.randrange(8)) if kc == 99 else draw(anc_under[node])
                            b0 = (b0[0], a[1]) if facet == "file" else ((a[0], b0[1]) if facet == "rank" else b0)
                            new = {(b0[0] + dx, b0[1] + dy): t for t, dx, dy in rel}
                            if all(0 <= x < 8 and 0 <= y < 8 and (x, y) not in board for x, y in new):
                                ok = new
                                break
                    else:
                        ((sq, t),) = pcs.items()
                        for _ in range(30):
                            if kc == 99:
                                s2 = (rng.randrange(8), rng.randrange(8))
                            elif kc == 0:
                                s2 = draw(sq_tok[t]) if t in sq_tok else sq
                            else:
                                s2 = draw(sq_all)
                            s2 = (s2[0], sq[1]) if facet == "file" else ((sq[0], s2[1]) if facet == "rank" else s2)
                            if s2 not in board:
                                t2 = t
                                if what and kc > 0:
                                    t2 = draw(tok_all if kc == 99 or s2 not in tok_sq else tok_sq[s2])
                                ok = {s2: t2}
                                break
                    if ok is None:
                        ok, kept = pcs, kept + 1
                    board.update(ok)
                out.append((board, kept, len(order)))
        return out

    for facet in ("file", "rank", "both"):
        for kc in KCS:
            run(f"where-{facet} kc={kc}", regen(kc, False, facet), N, res)
    for facet in ("file", "both"):
        for kc in KCS:
            run(f"where-{facet}+what kc={kc}", regen(kc, True, facet), N, res)
    items = list(raw_use)
    cum = list(itertools.accumulate(raw_use[i] for i in items))
    pitems = list(profiles)
    pcum = list(itertools.accumulate(profiles[i] for i in pitems))
    for counts in (0, 1):
        out = []
        for _ in range(len(held) * REP):
            board = {}
            if counts:
                for t, n in zip(KINDS, wchoice(pitems, pcum)):
                    for _ in range(n):
                        for _ in range(30):
                            tt, sq = wchoice(items, cum)
                            if tt == t and sq not in board:
                                board[sq] = t
                                break
            else:
                for _ in range(rng.choice(sizes)):
                    for _ in range(30):
                        t, sq = wchoice(items, cum)
                        if sq not in board:
                            board[sq] = t
                            break
            out.append((board, 0, 0))
        run(f"list counts={counts}", out, N, res)
    json.dump(res, open(OUTF, "w"), indent=1)
print("done", round(time.time() - t0), flush=True)
