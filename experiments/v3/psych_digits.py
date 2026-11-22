"""Skilled-memory digit span (subject DD, RSS95): Pilot B = decode accuracy against exposure, no staircase.

Design: docs/PLAN_PSYCH_DS.md (cited "plan §n"); human facts: docs/CE1981_PROTOCOL.md.  Pilot B (plan §6) tests the main risk R1
(span explodes because TRELLIS stores everything at once) and R2 (stall) before any full build.

Usage
  psych_digits.py pilotB <prior: off|on|x10> <seed> <max_units> [time_budget_s=0] [list_len=40] [tag=pilotB] [mode=rule|forced3|forced4|forced34]
    mode rule = the plan (recognition rule decides the groups); forced* = DIAGNOSTIC: groups of 3 / 4 / 3-or-4 imposed, no recognition test
  psych_digits.py diag|diag2|diag3 ...             (recognition-bits diagnostics used in the report)
  psych_digits.py summary [tag=pilotB]            (reads results/psych_digits/<tag>_*_s*.json, writes <tag>.md)
The full staircase simulation (plan §8) is to be added as another mode that reuses Learner / encode_list / decode below.

Learner: configuration C3 of the paper (confs/trellis-v2/paper/main.tex, Table "Configurations of TRELLIS": "Learner of the generation
study (posterior storage, window 7); default parser"), as in psych_segment.py: logs/v3_r212_grid.sh S + D0 + window 7.  Dropped:
story_facet (needs the previous sentence of a story).  Everything else unchanged: store="posterior" (NOT "best" as in the plan header
and as chess uses), refresh_budget 0, repair_path/repair_use 700, exact_alone "content", children/spine/heights False, naming "bands",
tree "cu", refresh "verify", restructure "sample", type_names False, word_cc False.  Chess uses store="best" with refresh_budget 300
(chess_cs1973_exact launcher); the plan's header names that configuration, so the plan and the C3 learner differ and the plan text must
be changed to whichever is chosen (see report).  Window 7 with isolated 3-4 token experiences means every context slot beyond the
group reads BOS/EOS.

Encoding (plan §2): auditory buffer B=4, isolated prefix parses of length 4 then 3, learning off (L._quiet = True, L.parse as
chess_cs1973_exact.View.__init__ l.388-392), recognised iff L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx) >= THETA (=16)
(View.bits l.397-401, View.walk l.403-406).  Label = (content concept, context concept) = ends of climb_path (l.375-385; the deepest
node on the sorting path with count >= L.support).  `_exp` is not needed (Trellis3, not GraphTrellis).

Decoding (chess View.decode l.414-433, Mem.composite l.350-363): identity-on = exact if the group string is already a stored chunk,
else the most frequent stored instance under (content concept, context concept) (instances: stored composites whose content leaf is
under the content concept and context leaf under the context concept; the vote is the record's posterior weight, which is 1 for
store="best" as in chess); identity-off = the concept-pair rule only; either concept at the root = omission.
Posterior storage changes "stored chunk": a composite string is a stored chunk when the summed posterior weight of its records is
>= IDENT_W = 0.5 (modelling choice, = story_chunks.py theta; shares at 0.25 and 1.0 are logged).  A unit learned as one experience
always stores its whole span with weight 1, so a group string seen as a unit is stored; a 3-digit sub-composite of a 4-digit unit
counts only if its summed posterior reaches 0.5, so sub-composite commitment is soft (store="best" would commit the chosen ones
at weight 1).

Both decode rules are scored on the same learner trajectory: neither rule changes what is recognised or learned (labels are fixed
during a trial and learning follows recall), so separate runs would reproduce the same trajectory.
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.trellis3 import Trellis3  # noqa: E402

OUT = os.path.join("experiments", "v3", "results", "psych_digits")
THETA = 16.0
B = 4                      # auditory buffer (plan §2, RSS95 p. 320)
IDENT_W = 0.5
THR_LOG = (0.25, 0.5, 1.0)

CFG = {"children": False, "spine": False, "heights": False, "naming": "bands", "tree": "cu", "refresh": "verify",
       "refresh_budget": 0, "store": "posterior", "restructure": "sample", "type_names": False, "word_cc": False,
       "repair_path": True, "repair_use": 700, "exact_alone": "content", "window": 7}
DROPPED = {"story_facet": "needs the previous sentence of a story (learn(follows=...)); no sentence-to-sentence context here"}


# ----------------------------------------------------------------------------------------------- prior stream (plan §5)
def _tn(rng, lo, hi):
    mu, sd = (lo + hi) / 2, (hi - lo) / 6
    while True:
        v = rng.gauss(mu, sd)
        if lo <= v <= hi:
            return v


def _time(rng, lo, hi, kind):
    """Digits of a running time in seconds (lo, hi); kind: 'mss' 'msst_p3' 'mss_msst' 'mmss' 'msst'."""
    v = _tn(rng, lo, hi)
    tenths = kind == "msst" or (kind == "msst_p3" and rng.random() < 0.3)
    if kind == "sst":
        return "%03d" % min(999, max(100, int(round(v * 10))))
    if tenths:
        t = int(round(v * 10))
        t = min(t, int(hi * 10))
        sec, ten = divmod(t, 10)
    else:
        sec, ten = min(int(round(v)), int(hi)), None
    m, s = divmod(sec, 60)
    out = "%d%02d" % (m, s)
    if kind == "mmss" or (kind == "mss_msst" and m >= 10):
        out = "%d%02d" % (m, s)
    return out + (str(ten) if ten is not None else "")


def draw_item(rng):
    """One prior-stream item: (category letter, digit string).  Ranges, shapes (truncated normal at the range centre, sd = range/6)
    and weights are modelling choices (plan §5), not fitted."""
    c = rng.choice("abcdefghijklm")
    if c == "a":
        return c, _time(rng, 45.0, 79.9, "sst")
    if c in "bcd":
        lo, hi = {"b": (110, 179), "c": (175, 269), "d": (225, 419)}[c]
        return c, _time(rng, lo, hi, "msst_p3")
    if c == "e":
        v = _tn(rng, 450, 719)
        sec = min(int(round(v)), 719)
        m, s = divmod(sec, 60)
        return c, "%d%02d" % (m, s)
    if c == "f":
        return c, _time(rng, 480, 599.9, "msst")
    if c == "g":
        return c, _time(rng, 600, 839, "mmss")
    if c == "h":
        return c, _time(rng, 780, 1199, "mmss")
    if c == "i":
        return c, _time(rng, 1620, 2999, "mmss")
    if c == "j":
        return c, _time(rng, 2760, 3599, "mmss")
    if c == "k":
        y = rng.randint(1900, 1989) if rng.random() < 0.7 else rng.randint(1492, 1899)
        return c, "%04d" % y
    def age():
        while True:
            a = int(round(rng.gauss(40, 18)))
            if 10 <= a <= 99:
                return a
    if c == "l":
        a = age()
        return c, "%02d" % a + (str(rng.randint(0, 9)) if rng.random() < 0.5 else "")
    return "m", "%02d%02d" % (age(), age())          # double ages


def prior_stream(seed, mult=1):
    """Smallest seeded stream whose running-time items (a-j) hold >= 500 distinct strings (RSS95 p. 318), times mult."""
    rng = random.Random(seed * 1000 + 7)
    items, distinct = [], set()
    while len(distinct) < 500:
        c, d = draw_item(rng)
        items.append((c, d))
        if c in "abcdefghij":
            distinct.add(d)
    n_pre = len(items)
    while len(items) < n_pre * mult:
        items.append(draw_item(rng))
    return items, n_pre


# ----------------------------------------------------------------------------------------------- memory, labels, decoding
def climb_path(L, tree, x):
    """x's sorting path, root first, up to its climbing ancestor (chess_cs1973_exact.climb_path l.375-385)."""
    path = tree.categorize(x)
    k = 0
    for j, n in enumerate(path[1:], 1):
        if n.count >= L.support:
            k = j
    return path[:k + 1]


def leaf_ids(tree, node):
    """Ids of the leaves under a concept (chess_cs1973_exact.leaf_ids l.313-333)."""
    cu = getattr(tree, "tree", None)
    out = []
    if cu is not None and hasattr(cu, "children_of"):
        stack = [node.id]
        while stack:
            i = stack.pop()
            kids = cu.children_of(i)
            if kids:
                stack.extend(kids)
            else:
                out.append(i)
        return out
    stack = [node]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            out.append(n.id)
    return out


class Mem:
    """Learner plus what the decoder reads: token strings of stored composites and their weights (chess `inst`, `store`)."""

    def __init__(self, seed):
        self.L = Trellis3(seed=seed, **CFG)
        self.comp_rids = []            # rids of stored composites
        self.inst = {}                 # rid -> token tuple
        self.span_w = defaultdict(float)   # token tuple -> summed posterior weight of its records
        self.n_exp = 0
        self.recs_of = defaultdict(list)   # token tuple -> rids of its composite records (P3c decoder)
        self.fam_pos = {3: defaultdict(set), 4: defaultdict(set)}   # P3c: familiar strings (span_w >= .5) by (position, digit)
        self.fam = {3: set(), 4: set()}

    def mark_fam(self, t):
        k = len(t)
        if k in self.fam and t not in self.fam[k] and self.span_w.get(t, 0.0) >= 0.5:
            self.fam[k].add(t)
            for i, d in enumerate(t):
                self.fam_pos[k][(i, d)].add(t)

    def learn(self, tokens):
        L = self.L
        L._quiet = False
        a0 = len(L.records)
        L.learn(list(tokens))
        self.n_exp += 1
        for e in L.records[a0:]:
            if e.token is None:
                t = tuple(tokens[e.span[0]:e.span[1]])
                self.inst[e.rid] = t
                self.comp_rids.append(e.rid)
                self.span_w[t] += e.weight
                self.recs_of[t].append(e.rid)
                self.mark_fam(t)

    def top(self, tokens):
        """Top composite of the isolated parse of tokens, learning off (chess View.__init__)."""
        L = self.L
        L._quiet = True
        try:
            r = L.parse(list(tokens))
        finally:
            L._quiet = False
        n = len(tokens)
        for k in r["chosen"]:
            if (k[0], k[2]) == (0, n):
                return r["comps"][k]
        return None

    def bits(self, e):
        L = self.L
        L._quiet = True
        try:
            return L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
        finally:
            L._quiet = False

    def index(self):
        """cnt_at: content leaf id -> composite records (chess Mem.__init__ l.340-348)."""
        L = self.L
        cnt_at = defaultdict(list)
        for rid in self.comp_rids:
            e = L.records[rid]
            if e.leaf is not None and e.cleaf is not None:
                cnt_at[e.cleaf.id].append(e)
        return cnt_at


class View:
    """Per-list decode helper (chess View.decode l.414-433 / Mem.composite l.350-363, for Trellis3 records)."""

    def __init__(self, mem):
        self.mem = mem
        self.cnt_at = mem.index()
        self.cache = {}

    def label(self, e):
        L = self.mem.L
        L._quiet = True
        try:
            pc = climb_path(L, L.cnt, L._content(e))
            px = climb_path(L, L.ctx, e.ctx)
        finally:
            L._quiet = False
        return pc, px

    def composite(self, nc, nx):
        L = self.mem.L
        key = (nc.id, nx.id)
        if key not in self.cache:
            xs = set(leaf_ids(L.ctx, nx))
            cnt, first = Counter(), {}
            for lid in leaf_ids(L.cnt, nc):
                for e in self.cnt_at.get(lid, ()):
                    if e.leaf.id in xs:
                        k = self.mem.inst[e.rid]
                        cnt[k] += e.weight
                        first.setdefault(k, k)
            self.cache[key] = min(cnt, key=lambda k: (-cnt[k], k)) if cnt else None
        return self.cache[key]

    def concept_decode(self, pc, px):
        if len(pc) > 1 and len(px) > 1:
            return self.composite(pc[-1], px[-1])
        return None


# ----------------------------------------------------------------------------------------------- encoding (plan §2)
def encode_list(mem, digits):
    """Units of one list: [("g", tokens, e, rs)] groups (rs = went into an RS slot, i.e. recognised while |A| = B in the stream; the
    tail flush at the end of the list goes to STM) and [("s", digit)] single digits (STM).  Learning off throughout."""
    A, units = [], []

    def attempt(flush):
        for size in (4, 3):
            if len(A) >= size:
                e = mem.top(A[:size])
                bb = mem.bits(e) if e is not None else -99.0
                mem.maxbits = max(getattr(mem, "maxbits", -99.0), bb)
                if e is not None and bb >= THETA:
                    units.append(("g", tuple(A[:size]), e, not flush))
                    del A[:size]
                    return True
        units.append(("s", A.pop(0)))
        return False

    for d in digits:
        A.append(d)
        if len(A) == B:
            attempt(False)
    while A:
        attempt(True)
    return units


def learn_pieces(units):
    """What is stored (plan §2): every group, and every run of ungrouped digits cut into pieces of B."""
    out, run = [], []
    for u in units + [("g", None, None, None)]:
        if u[0] == "s":
            run.append(u[1])
            continue
        for i in range(0, len(run), B):
            out.append(tuple(run[i:i + B]))
        run = []
        if u[1] is not None:
            out.append(u[1])
    return out


# ----------------------------------------------------------------------------------------------- Pilot B
def forced_units(mem, digits, rng, mode):
    """Diagnostic arm (NOT the plan rule): groups of the given sizes are imposed without the recognition test, so decode accuracy
    can be measured although recognition never fires (see report).  mode forced3 / forced4 / forced34 (sizes 3 or 4 at random)."""
    units, i, n = [], 0, len(digits)
    while i < n:
        rem = n - i
        opts = [s for s in ((3,) if mode == "forced3" else (4,) if mode == "forced4" else (3, 4)) if rem - s == 0 or rem - s >= 3]
        if not opts:
            units += [("s", d) for d in digits[i:]]
            break
        s = rng.choice(opts)
        tok = tuple(digits[i:i + s])
        units.append(("g", tok, mem.top(list(tok)), True))
        i += s
    return units


def pilot_b(prior, seed, max_units, budget, list_len, tag, mode="rule"):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    mem = Mem(seed)
    info = {"prior": prior, "seed": seed, "mode": mode}
    if prior != "off":
        items, n_pre = prior_stream(seed, 10 if prior == "x10" else 1)
        for c, d in items:
            mem.learn(list(d))
        info.update(n_pre=n_pre, prior_items=len(items), prior_seconds=round(time.time() - t0, 1),
                    prior_records=len(mem.L.records))
        print("prior", info, flush=True)
    rng = random.Random(seed * 31 + 5)
    rows, units_cum, groups_cum, lst = [], 0, 0, 0
    stop = "max_units"
    while units_cum < max_units:
        if budget and time.time() - t0 > budget:
            stop = "time_budget"
            break
        digits = [str(rng.randrange(10)) for _ in range(list_len)]
        mem.maxbits = -99.0
        units = encode_list(mem, digits) if mode == "rule" else forced_units(mem, digits, rng, mode)
        view = View(mem)
        row = {"n": units_cum, "grp": groups_cum, "len": list_len}
        for sz in (3, 4):
            for k in ("n", "idon", "idoff", "ident", "omit", "wrong", "stored_by_w"):
                row["%s%d" % (k, sz)] = 0
        row["stb25_4"] = row["stb100_4"] = row["stb25_3"] = row["stb100_3"] = 0
        row["flush_groups"] = 0
        row["digits_in_rs"] = 0
        row["single"] = 0
        row["maxbits"] = round(mem.maxbits, 2)
        for u in units:
            if u[0] == "s":
                row["single"] += 1
                continue
            _, tok, e, rs = u
            if not rs:
                row["flush_groups"] += 1
                continue
            sz = len(tok)
            w = mem.span_w.get(tok, 0.0)
            ident = w >= IDENT_W
            pc, px = view.label(e)
            if mode != "rule":
                bb = mem.bits(e)
                row["bits_sum"] = row.get("bits_sum", 0.0) + bb
                row["bits_max"] = max(row.get("bits_max", -99.0), bb)
                row["bits_ge"] = row.get("bits_ge", 0) + int(bb >= THETA)
            dec = view.concept_decode(pc, px)
            ok_c = dec == tok
            row["n%d" % sz] += 1
            row["digits_in_rs"] += sz
            row["idon%d" % sz] += int(ident or ok_c)
            row["idoff%d" % sz] += int(ok_c)
            row["ident%d" % sz] += int(ident)
            row["stored_by_w%d" % sz] += int(ident)
            row["stb25_%d" % sz] += int(w >= 0.25)
            row["stb100_%d" % sz] += int(w >= 1.0)
            if dec is None:
                row["omit%d" % sz] += 1
            elif not ok_c:
                row["wrong%d" % sz] += 1
        pieces = learn_pieces(units)
        row["pieces"] = len(pieces)
        row["pieces_group"] = sum(1 for u in units if u[0] == "g")
        for p in pieces:
            mem.learn(p)
        units_cum += len(pieces)
        groups_cum += row["n3"] + row["n4"]
        row["t"] = round(time.time() - t0, 1)
        row["recs"] = len(mem.L.records)
        rows.append(row)
        lst += 1
        if lst % 50 == 0:
            print("list %d units %d groups %d t %.0fs recs %d" % (lst, units_cum, groups_cum, row["t"], row["recs"]), flush=True)
            dump(tag, prior, seed, info, rows, units_cum, stop + "(partial)", t0, list_len)
    dump(tag, prior, seed, info, rows, units_cum, stop, t0, list_len)
    print("done", units_cum, "units", round(time.time() - t0), "s", stop, flush=True)


def dump(tag, prior, seed, info, rows, units_cum, stop, t0, list_len):
    base = {"experiment": "psych_digits pilot B", "cfg": CFG, "cfg_dropped": DROPPED, "theta": THETA, "buffer": B,
            "ident_weight": IDENT_W, "list_len": list_len, "info": info, "stopped": stop, "units": units_cum,
            "lists": len(rows), "seconds": round(time.time() - t0, 1),
            "note": "both decode rules are scored on one learner trajectory; rows = one list each", "rows": rows}
    for dec in ("idon", "idoff"):
        with open(os.path.join(OUT, "%s_%s_%s-%s_s%d.json" % (tag, info.get("mode", "rule"), prior, dec, seed)), "w", encoding="utf-8") as f:
            json.dump({**base, "decode": dec}, f)


# ----------------------------------------------------------------------------------------------- summary / projection
T_SESS = [(15, 25), (5, 15), (5, 10), (15, 8), (180, 6), (45, 5), (210, 4), (390, 3)]       # RSS95 Table 1 (P §2)


def lists_per_session(s):
    for cnt, t in T_SESS:
        if s <= cnt:
            return t
        s -= cnt
    return 3


def window_stats(rows, lo, hi, dec):
    tot = Counter()
    for r in rows:
        if lo < r["n"] <= hi:
            for sz in (3, 4):
                tot["n%d" % sz] += r["n%d" % sz]
                tot["c%d" % sz] += r["%s%d" % (dec, sz)]
                tot["ident%d" % sz] += r["ident%d" % sz]
                tot["omit%d" % sz] += r["omit%d" % sz]
                tot["wrong%d" % sz] += r["wrong%d" % sz]
            tot["digits_rs"] += r["digits_in_rs"]
            tot["digits"] += r["len"]
            tot["units"] += r["pieces"]
            tot["flush"] += r["flush_groups"]
            tot["single"] += r["single"]
    return tot


def p_of(c, n):
    return (c + 0.5) / (n + 1) if n else float("nan")


def project(rows, dec, sessions=865):
    """Staircase projection from p(n): span = s_bar ln2 / -ln p (plan §1), list length tracks the projected span, exposure per
    session = lists x span digits x (units per digit measured at that exposure)."""
    pts = []
    mx = rows[-1]["n"] + rows[-1]["pieces"]
    n = 250.0
    while n < mx * 1.0001:
        t = window_stats(rows, n / 2, n, dec)
        ng = t["n3"] + t["n4"]
        if ng >= 20 and t["digits"]:
            p = p_of(t["c3"] + t["c4"], ng)
            sbar = (3 * t["n3"] + 4 * t["n4"]) / ng
            span = sbar * math.log(2) / -math.log(p)
            pts.append((n, span, t["units"] / t["digits"]))
        n *= 1.15
    if not pts:
        return None
    ns = [x[0] for x in pts]

    def f(n, j):
        if n <= ns[0]:
            return pts[0][j]
        if n >= ns[-1]:
            return pts[-1][j]
        for i in range(1, len(ns)):
            if n <= ns[i]:
                a = (math.log(n) - math.log(ns[i - 1])) / (math.log(ns[i]) - math.log(ns[i - 1]))
                return pts[i - 1][j] * (1 - a) + pts[i][j] * a
    n, hit, trace = 0.0, {}, {}
    extrap = None
    for s in range(1, sessions + 1):
        span = max(7.0, min(208.0, f(max(n, ns[0]), 1)))
        for thr in (20, 40, 80, 104):
            if thr not in hit and span >= thr:
                hit[thr] = s
        if s in (15, 50, 100, 250, 500, 865):
            trace[s] = round(span, 1)
        if n > ns[-1] and extrap is None:
            extrap = s
        n += lists_per_session(s) * span * f(max(n, ns[0]), 2)
    return {"hit": hit, "span_at": trace, "extrapolated_from_session": extrap, "n_end": round(n), "max_measured": ns[-1],
            "curve": [(round(a), round(b, 1)) for a, b, _ in pts]}


def summary(tag):
    import glob
    res = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(OUT, "%s_*_s*.json" % tag))):
        d = json.load(open(fn, encoding="utf-8"))
        res[(d["info"].get("mode", "rule") + "/" + d["info"]["prior"], d["decode"])][d["info"]["seed"]] = d
    cps = [500, 1000, 2000, 5000, 10000, 15000]
    lines = ["# Pilot B: decode accuracy against exposure (generated by psych_digits.py summary)", "",
             "Exposure = cumulative learned encoding units (groups + ungrouped pieces); p = decode accuracy of RS groups, window (n/2, n]; "
             "mean over seeds [min-max]. Projected span = s_bar ln2/(-ln p), p smoothed (c+.5)/(n+1). Share = digits inside RS groups / digits.", ""]
    for (prior, dec), runs in sorted(res.items()):
        seeds = sorted(runs)
        lines += ["## prior %s, decode %s (seeds %s)" % (prior, dec, seeds), "",
                  "| units | p(3) | p(4) | p(all) | n groups | share of digits in RS groups | 3-digit share of groups | projected span |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in cps:
            vals = defaultdict(list)
            for s in seeds:
                d = runs[s]
                if d["units"] < c * 0.9:
                    continue
                t = window_stats(d["rows"], c / 2, c, dec)
                ng = t["n3"] + t["n4"]
                if not ng:
                    continue
                vals["p3"].append(p_of(t["c3"], t["n3"]))
                vals["p4"].append(p_of(t["c4"], t["n4"]))
                p = p_of(t["c3"] + t["c4"], ng)
                vals["p"].append(p)
                vals["ng"].append(ng)
                vals["share"].append(t["digits_rs"] / t["digits"])
                vals["s3"].append(t["n3"] / ng)
                sb = (3 * t["n3"] + 4 * t["n4"]) / ng
                vals["span"].append(sb * math.log(2) / -math.log(p))

            def m(k, fmt="%.3f"):
                v = [x for x in vals[k] if x == x]
                return (fmt % (sum(v) / len(v)) + " [" + fmt % min(v) + "-" + fmt % max(v) + "]") if v else "-"
            lines.append("| %d | %s | %s | %s | %s | %s | %s | %s |" % (c, m("p3"), m("p4"), m("p"), m("ng", "%.0f"), m("share"), m("s3"), m("span", "%.1f")))
        bl = []
        for s in seeds:
            rr = runs[s]["rows"]
            if "bits_sum" in rr[-1] or any("bits_sum" in r for r in rr):
                ngt = sum(r["n3"] + r["n4"] for r in rr)
                bl.append("seed %d: mean bits %.1f, max %.1f, share >= THETA %.4f" % (s, sum(r.get("bits_sum", 0) for r in rr) / max(ngt, 1), max(r.get("bits_max", -99) for r in rr), sum(r.get("bits_ge", 0) for r in rr) / max(ngt, 1)))
            else:
                bl.append("seed %d: max recognition bits over all attempted prefixes %.1f (THETA %g); groups recognised in total %d" % (s, max(r["maxbits"] for r in rr), THETA, sum(r["n3"] + r["n4"] + r["flush_groups"] for r in rr)))
        lines += ["Recognition bits: " + "; ".join(bl), ""]
        lines.append("")
        lines += ["Projected staircase (per seed): session reaching span 20 / 40 / 80 / 104; span at sessions 15, 50, 100, 250, 500, 865;"
                  " first session beyond measured exposure.", ""]
        for s in seeds:
            pr = project(runs[s]["rows"], dec)
            lines.append("- seed %d (%d units, stopped %s): %s" % (s, runs[s]["units"], runs[s]["stopped"], json.dumps(pr and {k: pr[k] for k in ("hit", "span_at", "extrapolated_from_session", "max_measured")})))
        lines.append("")
    path = os.path.join(OUT, "%s.md" % tag)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "pilotB":
        a = sys.argv
        pilot_b(a[2], int(a[3]), int(a[4]), float(a[5]) if len(a) > 5 else 0.0, int(a[6]) if len(a) > 6 else 40,
                a[7] if len(a) > 7 else "pilotB", a[8] if len(a) > 8 else "rule")
    elif sys.argv[1] == "summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else "pilotB")


def diag(prior, seed, n_units):
    """Recognition bits of stored vs random 3/4-digit strings after prior + n_units random pieces (R3 check)."""
    mem = Mem(seed)
    items, _ = prior_stream(seed, 1)
    if prior != "off":
        for c, d in items:
            mem.learn(list(d))
    rng = random.Random(5)
    for _ in range(n_units):
        mem.learn([str(rng.randrange(10)) for _ in range(4)])
    out = {}
    for name, strs in (("prior_strings", [d for c, d in items[:60] if len(d) in (3, 4)]),
                       ("random4", ["".join(str(rng.randrange(10)) for _ in range(4)) for _ in range(40)]),
                       ("random3", ["".join(str(rng.randrange(10)) for _ in range(3)) for _ in range(40)])):
        bs = []
        for s in strs:
            e = mem.top(list(s))
            if e is not None:
                L = mem.L
                bc, bx = L._climb(L.cnt, L._content(e)), L._climb(L.ctx, e.ctx)
                bs.append((round(bc, 1), round(bx, 1), mem.span_w.get(tuple(s), 0.0)))
        out[name] = bs[:12]
        print(name, "mean total bits %.1f max %.1f" % (sum(a + b for a, b, _ in bs) / max(len(bs), 1), max((a + b for a, b, _ in bs), default=0)), bs[:8], flush=True)
    c = Counter()
    for k, v in mem.span_w.items():
        c[len(k)] += 1
    print("composite string counts by length", dict(c), "records", len(mem.L.records))


if __name__ == "__main__" and sys.argv[1] == "diag":
    diag(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))


def diag2(seed, n_units):
    mem = Mem(seed)
    rng = random.Random(5)
    pcs = [[str(rng.randrange(10)) for _ in range(4)] for _ in range(n_units)]
    for p in pcs:
        mem.learn(p)
    L = mem.L
    for p in pcs[:3] + pcs[-3:]:
        t = tuple(p)
        rid = [r for r in mem.comp_rids if mem.inst[r] == t and L.records[r].span == (0, 4)][0]
        rec = L.records[rid]
        e = mem.top(p)
        print("STR", t, "w", rec.weight)
        print(" stored cnt", rec.cnt)
        print(" parse  cnt", L._content(e))
        print(" stored ctx", {k: v for k, v in (rec.ctx or {}).items()})
        print(" parse  ctx", e.ctx)
        for tree, x, nm in ((L.cnt, rec.cnt, "stored-cnt"), (L.cnt, L._content(e), "parse-cnt")):
            path = tree.categorize(x)
            print(" ", nm, "path counts", [n.count for n in path], "bits", round(L._climb(tree, x), 1))
        print(" leaf id", rec.cleaf.id, "count", rec.cleaf.count)


if __name__ == "__main__" and sys.argv[1] == "diag2":
    diag2(int(sys.argv[2]), int(sys.argv[3]))


def diag3(seed, n_bg):
    """Bits of a probe string learned k times amid n_bg random 4-digit pieces (how many repeats reach THETA?)."""
    mem = Mem(seed)
    rng = random.Random(5)
    probes = {3: list("428"), 4: list("7391")}
    for _ in range(n_bg):
        mem.learn([str(rng.randrange(10)) for _ in range(4)])
    done = 0
    for k in (0, 1, 3, 10, 30, 100):
        while done < k:
            for p in probes.values():
                mem.learn(p)
            for _ in range(5):
                mem.learn([str(rng.randrange(10)) for _ in range(4)])
            done += 1
        L = mem.L
        res = []
        for sz, p in probes.items():
            e = mem.top(p)
            res.append((sz, round(L._climb(L.cnt, L._content(e)), 1), round(L._climb(L.ctx, e.ctx), 1)))
        print("repeats", k, "(cnt bits, ctx bits)", res, flush=True)


if __name__ == "__main__" and sys.argv[1] == "diag3":
    diag3(int(sys.argv[2]), int(sys.argv[3]))


# ----------------------------------------------------------------------------------------------- P3b staircase (docs/PLAN_PSYCH.md "P3b pre-registration")
# Usage: psych_digits.py stair <arm: full|prioroff|nolimit|rehearsal> <seed> [max_sessions=865] [budget_s=0] [tag=P3b]
#        psych_digits.py stairsummary [tag=P3b]
# Choices on literal readings (also in P3b_SUMMARY.md): recognition = span_w[prefix] >= REC_W (0.5), prefix of 4 then 3, nothing else
# (no parse, no bits).  Unrecognised head digit -> STM single.  At the list end the buffer is flushed the same way and everything goes to
# STM (a recognised flush group is one STM unit).  A list is correct iff the STM holds <= K = 7 units (FIFO overflow drops the oldest);
# every RS group decodes by identity because recognition == stored (span_w >= .5), so decode errors by group size are 0 by construction;
# the size comparison therefore uses the MISS RATE of the buffer attempts (share of 4- / 3-digit prefix attempts not recognised) as proxy.
REC_W = 0.5
K_STM = 7
STORE_S = 8.0
PROBE_SESSIONS = (50, 100, 250, 500, 865)
CONS = "BCDFGHJKLMNPQRSTVWXZ"
ARMS = ("full", "prioroff", "nolimit", "rehearsal")


def encode_stair(mem, seq, st):
    """Same structure as encode_list (buffer B = 4, prefix 4 then 3), recognition = stored familiarity span_w >= REC_W.
    st: Counter of attempts / hits by size (in-stream attempts only)."""
    A, units = [], []

    def attempt(flush):
        for size in (4, 3):
            if len(A) >= size:
                tok = tuple(A[:size])
                if not flush:
                    st["att%d" % size] += 1
                if mem.span_w.get(tok, 0.0) >= REC_W:
                    if not flush:
                        st["hit%d" % size] += 1
                    units.append(("g", tok, None, not flush))
                    del A[:size]
                    return
        units.append(("s", A.pop(0)))

    for d in seq:
        A.append(d)
        if len(A) == B:
            attempt(False)
    while A:
        attempt(True)
    return units


def stm_units(units):
    """STM load: ungrouped digits plus tail-flush groups (one unit each); RS groups use no STM."""
    return sum(1 for u in units if u[0] == "s" or not u[3])


def stair(arm, seed, max_sessions, budget, tag):
    assert arm in ARMS
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    mem = Mem(seed)
    info = {"arm": arm, "seed": seed, "rec_w": REC_W, "K": K_STM, "store_s": STORE_S, "start": 7}
    if arm != "prioroff":
        items, n_pre = prior_stream(seed, 1)
        for c, d in items:
            mem.learn(list(d))                     # DD's pre-experiment knowledge: no clock
        info.update(n_pre=n_pre, prior_items=len(items), prior_seconds=round(time.time() - t0, 1), prior_records=len(mem.L.records))
    print("prior", info, flush=True)
    rng = random.Random(seed * 31 + 5)
    prng = random.Random(seed * 77 + 11)
    gain = 2.9 if arm == "rehearsal" else 1.0
    clock, length, lst = 0.0, 7, 0
    rows, sess_rows, probes = [], [], {}
    stop = "max_sessions"
    for sess in range(1, max_sessions + 1):
        if budget and time.time() - t0 > budget:
            stop = "time_budget"
            break
        sstart = length
        for _ in range(lists_per_session(sess)):
            digits = [str(rng.randrange(10)) for _ in range(length)]
            st = Counter()
            units = encode_stair(mem, digits, st)
            nstm = stm_units(units)
            correct = nstm <= K_STM
            g3 = sum(1 for u in units if u[0] == "g" and len(u[1]) == 3)
            g4 = sum(1 for u in units if u[0] == "g" and len(u[1]) == 4)
            clock += gain * length
            nlearn = nreinf = 0
            for p in learn_pieces(units):
                if arm == "nolimit":
                    mem.learn(list(p))
                    nlearn += 1
                elif mem.span_w.get(p, 0.0) >= REC_W:
                    mem.span_w[p] += 1.0
                    nreinf += 1
                elif clock >= STORE_S:
                    mem.learn(list(p))
                    clock -= STORE_S
                    nlearn += 1
            rows.append({"s": sess, "len": length, "ok": int(correct), "g3": g3, "g4": g4, "e3": 0, "e4": 0, "stm": nstm,
                         "att4": st["att4"], "hit4": st["hit4"], "att3": st["att3"], "hit3": st["hit3"],
                         "learned": nlearn, "reinf": nreinf, "clock": round(clock, 1), "recs": len(mem.L.records),
                         "t": round(time.time() - t0, 1)})
            length = max(1, length + (1 if correct else -1))
            lst += 1
        sess_rows.append({"s": sess, "start_len": sstart, "end_len": length})
        if sess in PROBE_SESSIONS:
            probes[sess] = letter_probe(mem, prng)
            print("probe", sess, probes[sess], flush=True)
        if sess % 10 == 0:
            r = rows[-1]
            blk = [x["len"] for x in rows if x["s"] > sess - 10]
            print("session %d lists %d len(mean last10 sess) %.1f next %d clock %.1f recs %d t %.0fs" % (
                sess, lst, sum(blk) / len(blk), length, clock, r["recs"], r["t"]), flush=True)
            stair_dump(tag, arm, seed, info, rows, sess_rows, probes, stop + "(partial)", t0, sess)
    stair_dump(tag, arm, seed, info, rows, sess_rows, probes, stop, t0, sess_rows[-1]["s"] if sess_rows else 0)
    print("done", len(sess_rows), "sessions", round(time.time() - t0), "s", stop, flush=True)


def letter_probe(mem, rng):
    """10-list staircase on random consonants from 5; no learning, no clock."""
    length, hist = 5, []
    for _ in range(10):
        seq = [rng.choice(CONS) for _ in range(length)]
        units = encode_stair(mem, seq, Counter())
        ok = stm_units(units) <= K_STM
        hist.append((length, int(ok)))
        length = max(1, length + (1 if ok else -1))
    return {"final": length, "max": max(h[0] for h in hist), "hist": hist}


def stair_dump(tag, arm, seed, info, rows, sess_rows, probes, stop, t0, sess):
    d = {"experiment": "psych_digits P3b staircase", "cfg": CFG, "info": info, "stopped": stop, "sessions": sess,
         "lists": len(rows), "seconds": round(time.time() - t0, 1), "probes": probes, "sess_rows": sess_rows, "rows": rows}
    fn = os.path.join(OUT, "%s_%s_s%d.json" % (tag, arm, seed))
    with open(fn + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f)
    os.replace(fn + ".tmp", fn)


def _span_at(d, s):
    """Span = mean list length over the 5-session block ending at session s (Fig. 4 style)."""
    v = [r["len"] for r in d["rows"] if s - 4 <= r["s"] <= s]
    return sum(v) / len(v) if v else None


def _hit(d, thr):
    last = d["sessions"]
    for s in range(5, last + 1):
        v = _span_at(d, s)
        if v is not None and v >= thr:
            return s
    return None


def _miss(d, lo, hi):
    c = Counter()
    for r in d["rows"]:
        if lo < r["s"] <= hi:
            for k in ("att4", "hit4", "att3", "hit3"):
                c[k] += r[k]
    m4 = 1 - c["hit4"] / c["att4"] if c["att4"] else float("nan")
    m3 = 1 - c["hit3"] / c["att3"] if c["att3"] else float("nan")
    return m4, m3


def stairsummary(tag):
    import glob
    runs = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(OUT, "%s_*_s*.json" % tag))):
        d = json.load(open(fn, encoding="utf-8"))
        if d.get("experiment") != "psych_digits P3b staircase":
            continue
        runs[d["info"]["arm"]][d["info"]["seed"]] = d

    def ms(vals, fmt="%.1f"):
        v = [x for x in vals if x is not None and x == x]
        return (fmt % (sum(v) / len(v)) + " [" + fmt % min(v) + "-" + fmt % max(v) + "]") if v else "-"
    pts = (15, 50, 100, 250, 500, 865)
    L = ["# P3b: digit-span staircase with an 8 s storage limit (generated by psych_digits.py stairsummary)", "",
         "Pre-registration: docs/PLAN_PSYCH.md, section P3b. Span at session s = mean list length over the 5-session block ending at s. "
         "Cells are mean [min-max] over seeds that reached that session; n = seeds. Partial runs (wall time) are marked.", "",
         "DD points (CE1981_PROTOCOL / RSS95): start 7-9; about 80 by session ~500; 104 at ~865.", "",
         "## Runs", ""]
    for arm in ARMS:
        for s, d in sorted(runs.get(arm, {}).items()):
            L.append("- %s seed %d: %d sessions, %d lists, %.0f s, stopped %s" % (arm, s, d["sessions"], d["lists"], d["seconds"], d["stopped"]))
    L += ["", "## Span at sessions", "", "| arm | n | " + " | ".join("s%d" % p for p in pts) + " |", "|---|---|" + "---|" * len(pts)]
    for arm in ARMS:
        rr = runs.get(arm, {})
        if not rr:
            continue
        cells = []
        for p in pts:
            ds = [d for d in rr.values() if d["sessions"] >= p]
            cells.append(ms([_span_at(d, p) for d in ds]) + (" (n=%d)" % len(ds) if len(ds) != len(rr) else ""))
        L.append("| %s | %d | %s |" % (arm, len(rr), " | ".join(cells)))
    L.append("| DD | - | ~7-9 | - | - | - | ~80 | 104 (at ~865) |")
    L += ["", "## Session when span first reaches a level (per seed; - = not reached within the run)", "",
          "| arm | seed | sessions run | 20 | 40 | 80 | 104 |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for s, d in sorted(runs.get(arm, {}).items()):
            L.append("| %s | %d | %d | %s |" % (arm, s, d["sessions"], " | ".join(str(_hit(d, t) or "-") for t in (20, 40, 80, 104))))
    # signature effects
    L += ["", "## Pre-registered signature effects (primary clock = arm `full`)", ""]
    full = runs.get("full", {})
    po = runs.get("prioroff", {})

    def complete(rr):
        return rr and all(d["sessions"] >= 865 for d in rr.values())
    if full:
        mx = [max((_span_at(d, s) or 0) for s in range(5, d["sessions"] + 1)) for d in full.values()]
        e1 = "holds" if min(mx) >= 15 else ("fails" if complete(full) else "not reached") if max(mx) < 15 else "holds in some seeds"
        L.append("1. Span rises well above 7 (criterion: block span >= 15): max block span %s -> **%s**." % (ms(mx), e1))
        h80 = [_hit(d, 80) for d in full.values()]
        if all(h is not None and h >= 250 for h in h80):
            e2 = "holds (all seeds in [250, 865])"
        elif any(h is not None and h < 250 for h in h80):
            e2 = "fails (explosion: 80 before session 250 in %d seed(s))" % sum(1 for h in h80 if h is not None and h < 250)
        elif complete(full):
            e2 = "fails (stall: 80 never reached in 865 sessions)"
        else:
            e2 = "not reached (runs ended at session %s without 80)" % ",".join(str(d["sessions"]) for d in full.values())
        L.append("2. Span ~80 near session 500, not before 250, not never: first session at 80 = %s -> **%s**." % (h80, e2))
        m = []
        for d in full.values():
            hi = d["sessions"]
            m.append(_miss(d, max(0, hi - 50), hi))
        m4 = [x[0] for x in m if x[0] == x[0]]
        m3 = [x[1] for x in m if x[1] == x[1]]
        if m4 and m3:
            e3 = "holds" if sum(m4) / len(m4) > sum(m3) / len(m3) else "fails"
        else:
            e3 = "not reached"
        L.append("3. 4-digit groups err more than 3-digit groups. Decode errors are 0 by construction (recognised == stored, identity decode), "
                 "so the proxy is the miss rate of buffer attempts over the last 50 sessions run: 4-digit %s vs 3-digit %s -> **%s**." % (
                     ms(m4, "%.3f"), ms(m3, "%.3f"), e3))
        if po:
            rows = []
            res = []
            for p in (15, 50):
                a = [_span_at(d, p) for d in full.values() if d["sessions"] >= p]
                b = [_span_at(d, p) for d in po.values() if d["sessions"] >= p]
                if a and b:
                    rows.append("s%d full %.1f vs prioroff %.1f" % (p, sum(a) / len(a), sum(b) / len(b)))
                    res.append(sum(a) / len(a) > sum(b) / len(b))
            h = "; first-20 full %s vs prioroff %s" % ([_hit(d, 20) for d in full.values()], [_hit(d, 20) for d in po.values()])
            e4 = "not reached" if not res else ("holds" if all(res) else "fails" if not any(res) else "mixed")
            L.append("4. Prior knowledge speeds early learning: %s%s -> **%s**." % ("; ".join(rows), h, e4))
        else:
            L.append("4. Prior knowledge speeds early learning: no prioroff runs -> **not reached**.")
        fin = [pr["final"] for d in full.values() for pr in d["probes"].values()]
        mxl = [pr["max"] for d in full.values() for pr in d["probes"].values()]
        if fin:
            e5 = "holds" if sum(mxl) / len(mxl) <= 9 else "fails"
            L.append("5. Letter span stays about 7 (criterion: mean probe max length <= 9): final %s, max %s -> **%s**." % (ms(fin), ms(mxl), e5))
        else:
            L.append("5. Letter span stays about 7: no probe reached -> **not reached**.")
    L += ["", "## Letter probes (10-list staircase from 5, no learning): final / max length per probe session", "",
          "| arm | " + " | ".join("s%d" % p for p in PROBE_SESSIONS) + " |", "|---|" + "---|" * len(PROBE_SESSIONS)]
    for arm in ARMS:
        rr = runs.get(arm, {})
        if not rr:
            continue
        cells = []
        for p in PROBE_SESSIONS:
            v = [d["probes"][str(p)] for d in rr.values() if str(p) in d["probes"]]
            cells.append("final %s / max %s" % (ms([x["final"] for x in v]), ms([x["max"] for x in v])) if v else "-")
        L.append("| %s | %s |" % (arm, " | ".join(cells)))
    L += ["", "## 4- vs 3-digit groups: miss rate of buffer attempts over the 50 sessions before each probe (decode errors are 0 by construction)", "",
          "| arm | " + " | ".join("s%d" % p for p in PROBE_SESSIONS) + " |", "|---|" + "---|" * len(PROBE_SESSIONS)]
    for arm in ARMS:
        rr = runs.get(arm, {})
        if not rr:
            continue
        cells = []
        for p in PROBE_SESSIONS:
            v = [_miss(d, p - 50, p) for d in rr.values() if d["sessions"] >= p]
            cells.append("4: %s / 3: %s" % (ms([x[0] for x in v], "%.3f"), ms([x[1] for x in v], "%.3f"))) if v else cells.append("-")
        L.append("| %s | %s |" % (arm, " | ".join(cells)))
    L += ["", "## Choices on ambiguous points", "",
          "- Recognition: span_w[tuple] >= 0.5 for the 4-digit prefix, else the 3-digit prefix; no parse. The prior stream is learned without a clock.",
          "- A list is correct iff the STM holds at most 7 units: ungrouped digits plus tail-flush groups (one unit per recognised flush group). "
          "RS groups use no STM. Because recognition equals stored (span_w >= 0.5), identity decoding of an RS group always succeeds; "
          "the 4-vs-3 comparison therefore uses the miss rate of buffer attempts (a proxy, not a decode error).",
          "- Clock: +1 s (rehearsal arm +2.9 s) per digit presented, added before learning, carried across lists and sessions, starts at 0. "
          "Pieces from learn_pieces in list order; a piece already at span_w >= 0.5 gets +1 for free (no learn()); otherwise learn() costs 8 s if the clock holds >= 8 s. "
          "Reinforcement is checked against the weight at the moment the piece is reached (earlier pieces of the same list may already have raised it).",
          "- Staircase: starts at 7, +1 / -1, floor 1, no cap; length carries over between sessions. Span = mean list length in the 5-session block ending at s.",
          "- Letter probes: consonants never seen, so nothing is recognised; every letter goes to the STM (expected span ~7). Probe lists draw from a separate RNG.",
          "- No resume after a wall-time stop: the learner is not serialised; the JSON checkpoint every 10 sessions is the result.", ""]
    path = os.path.join(OUT, "%s_SUMMARY.md" % tag)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


# ----------------------------------------------------------------------------------------------- P3c: retrieval failure + label rescue
# Pre-registration: docs/PLAN_PSYCH.md "P3c".  P3b unchanged plus: each RS group's slot holds its digits and its TRELLIS label
# (c = deepest supported content concept, x = deepest supported context concept of the isolated parse, read before this list is
# learned).  Each digit and each label part is lost once with probability F (EPAM IV, RSS95 pp. 315, 319).  Recall: all digits
# intact -> recalled; else the best familiar string of length k agreeing with the surviving digits, scored by its record weight under
# the surviving label parts (chess composite decoder + partial image), or by span_w if no label part survives.
F_ARMS = {"full": dict(f=0.25), "norescue": dict(f=0.25, rescue=False), "ctxoff": dict(f=0.25, ctx=False),
          "prioroff": dict(f=0.25, prior=False), "f20": dict(f=0.20), "f30": dict(f=0.30)}


class Decoder:
    """Per-list label reader and partial-image decoder (memory is frozen during a list: learning comes after recall)."""

    def __init__(self, mem):
        self.mem = mem
        self.view = View.__new__(View)
        self.view.mem = mem
        self.labels = {}
        self.leafsets = {}

    def label(self, g):
        if g not in self.labels:
            e = self.mem.top(list(g))
            c = x = None
            if e is not None:
                pc, px = self.view.label(e)
                c = pc[-1] if len(pc) > 1 else None
                x = px[-1] if len(px) > 1 else None
            self.labels[g] = (c, x)
        return self.labels[g]

    def leaves(self, tree, node):
        key = (id(tree), node.id)
        if key not in self.leafsets:
            self.leafsets[key] = set(leaf_ids(tree, node))
        return self.leafsets[key]

    def decode(self, k, known, c, x):
        """known: {position: digit} surviving; c / x: surviving label parts (or None).  Returns the recalled string or None."""
        mem, L = self.mem, self.mem.L
        if known:
            sets = sorted((mem.fam_pos[k].get((i, d), set()) for i, d in known.items()), key=len)
            cand = set(sets[0])
            for s in sets[1:]:
                cand &= s
        else:
            cand = mem.fam[k]
        if not cand:
            return None
        if c is None and x is None:
            return min(cand, key=lambda t: (-mem.span_w.get(t, 0.0), t))
        lc = self.leaves(L.cnt, c) if c is not None else None
        lx = self.leaves(L.ctx, x) if x is not None else None
        best, bw = None, 0.0
        for t in cand:
            w = 0.0
            for rid in mem.recs_of.get(t, ()):
                e = L.records[rid]
                if e.leaf is None or e.cleaf is None:
                    continue
                if lc is not None and e.cleaf.id not in lc:
                    continue
                if lx is not None and e.leaf.id not in lx:
                    continue
                w += e.weight
            if w > 0 and (w > bw or (w == bw and t < best)):
                best, bw = t, w
        return best


def recall_groups(dec, units, frng, f, rescue, use_ctx, tally):
    """True iff every RS group is recalled.  tally: Counter of per-size / per-route counts."""
    ok_all = True
    for u in units:
        if u[0] != "g" or not u[3]:
            continue
        g = u[1]
        k = len(g)
        c, x = dec.label(g)
        if not use_ctx:
            x = None
        known = {i: d for i, d in enumerate(g) if frng.random() >= f}
        c = c if (c is not None and frng.random() >= f) else None
        x = x if (x is not None and frng.random() >= f) else None
        tally["n%d" % k] += 1
        if len(known) == k:
            tally["intact%d" % k] += 1
            continue
        route = ("c" if c is not None else "") + ("x" if x is not None else "") or "d"
        tally["try_" + route] += 1
        got = dec.decode(k, known, c, x) if rescue else None
        if got == g:
            tally["resc_" + route] += 1
        else:
            tally["err%d" % k] += 1
            ok_all = False
    return ok_all


def forget(arm, seed, max_sessions, budget, tag):
    opt = dict(f=0.25, rescue=True, ctx=True, prior=True)
    opt.update(F_ARMS[arm])
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    mem = Mem(seed)
    info = {"arm": arm, "seed": seed, "rec_w": REC_W, "K": K_STM, "store_s": STORE_S, "start": 7, **opt}
    if opt["prior"]:
        items, n_pre = prior_stream(seed, 1)
        for c, d in items:
            mem.learn(list(d))
        info.update(n_pre=n_pre, prior_items=len(items), prior_seconds=round(time.time() - t0, 1), prior_records=len(mem.L.records))
    print("prior", info, flush=True)
    rng = random.Random(seed * 31 + 5)          # same list stream as P3b
    prng = random.Random(seed * 77 + 11)
    frng = random.Random(seed * 13 + 7)         # forgetting draws
    clock, length, lst = 0.0, 7, 0
    rows, sess_rows, probes = [], [], {}
    stop = "max_sessions"
    for sess in range(1, max_sessions + 1):
        if budget and time.time() - t0 > budget:
            stop = "time_budget"
            break
        sstart = length
        for _ in range(lists_per_session(sess)):
            digits = [str(rng.randrange(10)) for _ in range(length)]
            st = Counter()
            units = encode_stair(mem, digits, st)
            nstm = stm_units(units)
            tally = Counter()
            dec = Decoder(mem)
            rs_ok = recall_groups(dec, units, frng, opt["f"], opt["rescue"], opt["ctx"], tally)
            correct = nstm <= K_STM and rs_ok
            clock += length
            nlearn = nreinf = 0
            for p in learn_pieces(units):
                if mem.span_w.get(p, 0.0) >= REC_W:
                    mem.span_w[p] += 1.0
                    nreinf += 1
                elif clock >= STORE_S:
                    mem.learn(list(p))
                    clock -= STORE_S
                    nlearn += 1
            row = {"s": sess, "len": length, "ok": int(correct), "stm_ok": int(nstm <= K_STM), "rs_ok": int(rs_ok), "stm": nstm,
                   "att4": st["att4"], "hit4": st["hit4"], "att3": st["att3"], "hit3": st["hit3"],
                   "learned": nlearn, "reinf": nreinf, "clock": round(clock, 1), "recs": len(mem.L.records), "t": round(time.time() - t0, 1)}
            row.update({k: v for k, v in tally.items()})
            rows.append(row)
            length = max(1, length + (1 if correct else -1))
            lst += 1
        sess_rows.append({"s": sess, "start_len": sstart, "end_len": length})
        if sess in PROBE_SESSIONS:
            probes[sess] = letter_probe(mem, prng)
            print("probe", sess, probes[sess], flush=True)
        if sess % 10 == 0:
            r = rows[-1]
            blk = [x["len"] for x in rows if x["s"] > sess - 10]
            print("session %d lists %d len(mean last10 sess) %.1f next %d clock %.1f recs %d t %.0fs" % (
                sess, lst, sum(blk) / len(blk), length, clock, r["recs"], r["t"]), flush=True)
            forget_dump(tag, arm, seed, info, rows, sess_rows, probes, stop + "(partial)", t0, sess)
    forget_dump(tag, arm, seed, info, rows, sess_rows, probes, stop, t0, sess_rows[-1]["s"] if sess_rows else 0)
    print("done", len(sess_rows), "sessions", round(time.time() - t0), "s", stop, flush=True)


def forget_dump(tag, arm, seed, info, rows, sess_rows, probes, stop, t0, sess):
    d = {"experiment": "psych_digits P3c forgetting", "cfg": CFG, "info": info, "stopped": stop, "sessions": sess,
         "lists": len(rows), "seconds": round(time.time() - t0, 1), "probes": probes, "sess_rows": sess_rows, "rows": rows}
    fn = os.path.join(OUT, "%s_%s_s%d.json" % (tag, arm, seed))
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    os.replace(fn + ".tmp", fn)


def _sum_rows(d, lo, hi, keys, minlen=0):
    c = Counter()
    for r in d["rows"]:
        if lo < r["s"] <= hi and r["len"] >= minlen:
            for k in keys:
                c[k] += r.get(k, 0)
    return c


def forgetsummary(tag):
    import glob
    runs = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(OUT, "%s_*_s*.json" % tag))):
        d = json.load(open(fn, encoding="utf-8"))
        if d.get("experiment") != "psych_digits P3c forgetting":
            continue
        runs[d["info"]["arm"]][d["info"]["seed"]] = d

    def ms(vals, fmt="%.1f"):
        v = [x for x in vals if x is not None and x == x]
        return (fmt % (sum(v) / len(v)) + " [" + fmt % min(v) + "-" + fmt % max(v) + "]") if v else "-"
    arms = list(F_ARMS)
    pts = (15, 50, 100, 250, 500, 865)
    L = ["# P3c: retrieval failure (EPAM IV 25%) with TRELLIS label rescue (generated by psych_digits.py forgetsummary)", "",
         "Pre-registration: docs/PLAN_PSYCH.md, section P3c. Span at session s = mean list length over the 5-session block ending at s; "
         "mean [min-max] over seeds that reached s.", "", "DD: start 7-9; about 80 by session ~500; 104 at ~865.", "", "## Runs", ""]
    for arm in arms:
        for s, d in sorted(runs.get(arm, {}).items()):
            L.append("- %s seed %d: %d sessions, %d lists, %.0f s, stopped %s" % (arm, s, d["sessions"], d["lists"], d["seconds"], d["stopped"]))
    L += ["", "## Span at sessions", "", "| arm | n | " + " | ".join("s%d" % p for p in pts) + " | max block span |", "|---|---|" + "---|" * (len(pts) + 1)]
    for arm in arms:
        rr = runs.get(arm, {})
        if not rr:
            continue
        cells = []
        for p in pts:
            ds = [d for d in rr.values() if d["sessions"] >= p]
            cells.append(ms([_span_at(d, p) for d in ds]) + (" (n=%d)" % len(ds) if len(ds) != len(rr) else ""))
        mx = [max((_span_at(d, s) or 0) for s in range(5, d["sessions"] + 1)) for d in rr.values()]
        L.append("| %s | %d | %s | %s |" % (arm, len(rr), " | ".join(cells), ms(mx)))
    L.append("| DD | - | ~7-9 | - | - | - | ~80 | 104 (at ~865) | 104 |")
    L += ["", "## Session when span first reaches a level (- = not reached)", "", "| arm | seed | sessions run | 20 | 40 | 80 | 104 |",
          "|---|---|---|---|---|---|---|"]
    for arm in arms:
        for s, d in sorted(runs.get(arm, {}).items()):
            L.append("| %s | %d | %d | %s |" % (arm, s, d["sessions"], " | ".join(str(_hit(d, t) or "-") for t in (20, 40, 80, 104))))
    keys = ["n3", "n4", "intact3", "intact4", "err3", "err4", "try_cx", "resc_cx", "try_c", "resc_c", "try_x", "resc_x", "try_d", "resc_d"]
    L += ["", "## Per-group recall by session window (pooled over seeds): reliability p = 1 - errors / groups; rescue share by route", "",
          "| arm | window | groups | p(3) | p(4) | digits intact | rescued cx | rescued c | rescued x | rescued digits-only |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        rr = runs.get(arm, {})
        for lo, hi in ((0, 50), (50, 100), (100, 250), (250, 500), (500, 865)):
            c = Counter()
            for d in rr.values():
                c.update(_sum_rows(d, lo, hi, keys))
            n = c["n3"] + c["n4"]
            if not n:
                continue
            fr = lambda a, b: ("%.3f (%d)" % (c[a] / c[b], c[b])) if c[b] else "-"
            p3 = "%.3f" % (1 - c["err3"] / c["n3"]) if c["n3"] else "-"
            p4 = "%.3f" % (1 - c["err4"] / c["n4"]) if c["n4"] else "-"
            L.append("| %s | %d-%d | %d | %s | %s | %.3f | %s | %s | %s | %s |" % (arm, lo, hi, n, p3, p4, (c["intact3"] + c["intact4"]) / n,
                     fr("resc_cx", "try_cx"), fr("resc_c", "try_c"), fr("resc_x", "try_x"), fr("resc_d", "try_d")))
    L += ["", "## Pre-registered signature effects (arm full)", ""]
    full, po, nr = runs.get("full", {}), runs.get("prioroff", {}), runs.get("norescue", {})

    def complete(rr):
        return bool(rr) and all(d["sessions"] >= 865 for d in rr.values())
    if full:
        mx = [max((_span_at(d, s) or 0) for s in range(5, d["sessions"] + 1)) for d in full.values()]
        L.append("1. Span rises well above 7 (block span >= 15): max %s -> **%s**." % (ms(mx), "holds" if min(mx) >= 15 else "fails" if max(mx) < 15 else "some seeds"))
        h80 = [_hit(d, 80) for d in full.values()]
        if all(h is not None and 250 <= h for h in h80):
            e2 = "holds"
        elif any(h is not None and h < 250 for h in h80):
            e2 = "fails (explosion)"
        elif complete(full):
            e2 = "fails (stall: 80 never reached)"
        else:
            e2 = "not reached (runs ended at %s)" % [d["sessions"] for d in full.values()]
        L.append("2. Rate (decision): first session at 80 = %s -> **%s**." % (h80, e2))
        c = Counter()
        for d in full.values():
            c.update(_sum_rows(d, 0, 865, keys, minlen=30))
        if c["n3"] and c["n4"]:
            r3, r4 = c["err3"] / c["n3"], c["err4"] / c["n4"]
            L.append("3. 4-digit groups err more (lists >= 30): 4-digit %.3f (%d groups) vs 3-digit %.3f (%d) -> **%s**." % (
                r4, c["n4"], r3, c["n3"], "holds" if r4 > r3 else "fails"))
        else:
            L.append("3. 4-digit groups err more: no lists >= 30 with both sizes -> **not reached**.")
        if po:
            res, rows = [], []
            for p in (15, 50):
                a = [_span_at(d, p) for d in full.values() if d["sessions"] >= p]
                b = [_span_at(d, p) for d in po.values() if d["sessions"] >= p]
                if a and b:
                    rows.append("s%d full %.1f vs prioroff %.1f" % (p, sum(a) / len(a), sum(b) / len(b)))
                    res.append(sum(a) / len(a) > sum(b) / len(b))
            L.append("4. Prior knowledge speeds early learning: %s -> **%s**." % ("; ".join(rows), "holds" if res and all(res) else "fails" if res and not any(res) else "mixed/not reached"))
        fin = [pr["final"] for d in full.values() for pr in d["probes"].values()]
        mxl = [pr["max"] for d in full.values() for pr in d["probes"].values()]
        if fin:
            L.append("5. Letter span about 7 (mean probe max <= 9): final %s, max %s -> **%s**." % (ms(fin), ms(mxl), "holds" if sum(mxl) / len(mxl) <= 9 else "fails"))
        if nr:
            a = [_span_at(d, 250) for d in full.values() if d["sessions"] >= 250]
            b = [_span_at(d, 250) for d in nr.values() if d["sessions"] >= 250]
            if a and b:
                L.append("Redundancy: span at s250 full %.1f vs no rescue %.1f -> **%s**." % (sum(a) / len(a), sum(b) / len(b), "holds" if sum(a) / len(a) > sum(b) / len(b) else "fails"))
    L += ["", "## Letter probes: final / max length", "", "| arm | " + " | ".join("s%d" % p for p in PROBE_SESSIONS) + " |", "|---|" + "---|" * len(PROBE_SESSIONS)]
    for arm in arms:
        rr = runs.get(arm, {})
        if rr:
            cells = []
            for p in PROBE_SESSIONS:
                v = [d["probes"][str(p)] for d in rr.values() if str(p) in d["probes"]]
                cells.append("%s / %s" % (ms([x["final"] for x in v]), ms([x["max"] for x in v])) if v else "-")
            L.append("| %s | %s |" % (arm, " | ".join(cells)))
    path = os.path.join(OUT, "%s_SUMMARY.md" % tag)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__" and sys.argv[1] == "forget":
    a = sys.argv
    forget(a[2], int(a[3]), int(a[4]) if len(a) > 4 else 865, float(a[5]) if len(a) > 5 else 0.0, a[6] if len(a) > 6 else "P3c")
if __name__ == "__main__" and sys.argv[1] == "forgetsummary":
    forgetsummary(sys.argv[2] if len(sys.argv) > 2 else "P3c")
if __name__ == "__main__" and sys.argv[1] == "stair":
    a = sys.argv
    stair(a[2], int(a[3]), int(a[4]) if len(a) > 4 else 865, float(a[5]) if len(a) > 5 else 0.0, a[6] if len(a) > 6 else "P3b")
if __name__ == "__main__" and sys.argv[1] == "stairsummary":
    stairsummary(sys.argv[2] if len(sys.argv) > 2 else "P3b")


# ----------------------------------------------------------------------------------------------- P3c rescue diagnostic (2026-11-21)
# Usage: psych_digits.py rescuediag <seed> [max_sessions=150]
# Re-runs the P3c full arm (same RNG streams, same trajectory) and, for every damaged RS group, logs why rescue fails:
# candidate count after survivors, after the content label; rank of the target by record weight; and what an episodic-recency
# rule (most recently encoded candidate wins; this list's groups are the most recent) would have picked.  No change to the run.
def rescuediag(seed, max_sessions):
    mem = Mem(seed)
    items, _ = prior_stream(seed, 1)
    for c, d in items:
        mem.learn(list(d))
    rng = random.Random(seed * 31 + 5)
    frng = random.Random(seed * 13 + 7)
    clock, length, lst = 0.0, 7, 0
    last = {}                                  # string -> (session, list index) of the last time it was encoded as a group
    out = []
    for sess in range(1, max_sessions + 1):
        for li in range(lists_per_session(sess)):
            digits = [str(rng.randrange(10)) for _ in range(length)]
            units = encode_stair(mem, digits, Counter())
            nstm = stm_units(units)
            dec = Decoder(mem)
            here = [u[1] for u in units if u[0] == "g"]
            for g in here:
                last[g] = (sess, lst)
            ok_all = True
            for u in units:
                if u[0] != "g" or not u[3]:
                    continue
                g = u[1]
                k = len(g)
                c, x = dec.label(g)
                known = {i: d for i, d in enumerate(g) if frng.random() >= 0.25}
                c = c if (c is not None and frng.random() >= 0.25) else None
                x = x if (x is not None and frng.random() >= 0.25) else None
                if len(known) == k:
                    continue
                got = dec.decode(k, known, c, x)
                ok = got == g
                ok_all &= ok
                if known:
                    sets = sorted((mem.fam_pos[k].get((i, d), set()) for i, d in known.items()), key=len)
                    cand = set(sets[0])
                    for s_ in sets[1:]:
                        cand &= s_
                else:
                    cand = set(mem.fam[k])
                ncand_c = None
                if c is not None:
                    lc = dec.leaves(mem.L.cnt, c)
                    ncand_c = sum(1 for t in cand if any(mem.L.records[r].cleaf is not None and mem.L.records[r].cleaf.id in lc
                                                         for r in mem.recs_of.get(t, ())))
                same_list = [t for t in cand if t != g and t in here]
                same_sess = [t for t in cand if t != g and t in last and last[t][0] == sess]
                rec_pick = max(cand, key=lambda t: (last.get(t, (-1, -1)), -mem.span_w.get(t, 0.0))) if cand else None
                rank = 1 + sum(1 for t in cand if mem.span_w.get(t, 0.0) > mem.span_w.get(g, 0.0))
                out.append({"s": sess, "k": k, "miss": k - len(known), "c": c is not None, "x": x is not None, "ok": ok,
                            "in": g in cand, "n": len(cand), "nc": ncand_c, "rank_w": rank, "same_list": len(same_list),
                            "same_sess": len(same_sess), "rec_ok": rec_pick == g, "len": length})
            correct = nstm <= K_STM and ok_all
            clock += length
            for p in learn_pieces(units):
                if mem.span_w.get(p, 0.0) >= REC_W:
                    mem.span_w[p] += 1.0
                elif clock >= STORE_S:
                    mem.learn(list(p))
                    clock -= STORE_S
            length = max(1, length + (1 if correct else -1))
            lst += 1
        if sess % 25 == 0:
            print("session", sess, "len", length, "attempts", len(out), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rescuediag_s%d.json" % seed), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    def rep(rows, name):
        if not rows:
            return
        n = len(rows)
        f = lambda key: sum(1 for r in rows if r[key]) / n
        med = lambda key: sorted(r[key] for r in rows)[n // 2]
        print("%-28s n %5d ok %.2f in_cand %.2f rec_ok %.2f | med cand %d, med rank_w %d, share same-list comp %.2f, same-sess comp %.2f"
              % (name, n, f("ok"), f("in"), f("rec_ok"), med("n"), med("rank_w"),
                 sum(1 for r in rows if r["same_list"]) / n, sum(1 for r in rows if r["same_sess"]) / n))
    for lo, hi in ((1, 50), (51, 100), (101, max_sessions)):
        w = [r for r in out if lo <= r["s"] <= hi]
        print("== sessions %d-%d" % (lo, hi))
        rep(w, "all")
        for m in (1, 2, 3):
            rep([r for r in w if r["miss"] == m], "missing %d" % m)
        rep([r for r in w if r["c"]], "content label survives")
        rep([r for r in w if not r["c"] and not r["x"]], "no label")
        cs = [r for r in w if r["c"] and r["nc"] is not None]
        if cs:
            print("   median candidates under surviving content label: %d (vs %d before)" % (
                sorted(r["nc"] for r in cs)[len(cs) // 2], sorted(r["n"] for r in cs)[len(cs) // 2]))


if __name__ == "__main__" and sys.argv[1] == "rescuediag":
    rescuediag(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 150)


# ----------------------------------------------------------------------------------------------- P3d (docs/PLAN_PSYCH.md, section P3d)
# Usage: psych_digits.py epam <arm: full|actoff|conceptsoff|prioroff|f0> <seed> [max_sessions=865] [budget_s=0] [tag=P3d]
#        psych_digits.py epamsummary [tag=P3d]
# EPAM IV's architecture with TRELLIS as the semantic net: groups from EPAM's retrieval-structure patterns (obligatory slots),
# rehearsal of the last <= 7 digits, storage during presentation at 8 s per new chunk (clock reset per list), activation for one
# session, 25% forgetting of slot items, rescue among activated familiar strings guided by the label.
D_ARMS = {"full": {}, "actoff": dict(act=False), "conceptsoff": dict(concepts=False), "prioroff": dict(prior=False), "f0": dict(f=0.0)}
D_REHEARSE = 7


def rs_sizes(n):
    """Group sizes for an n-item list (plan P3d step 1) and the number of trailing singles."""
    sizes, r = [], n
    if r >= 16:
        sizes += [4, 4, 4, 4]
        r -= 16
        while r >= 21:
            sizes += [3, 3, 3, 4, 4, 4]
            r -= 21
    m = r % 3
    if m == 0:
        sizes += [3] * (r // 3)
        return sizes, 0
    if m == 1 and r >= 4:
        sizes += [3] * ((r - 4) // 3) + [4]
        return sizes, 0
    if m == 2 and r >= 8:
        sizes += [3] * ((r - 8) // 3) + [4, 4]
        return sizes, 0
    sizes += [3] * (r // 3)                      # r = 1, 2, 5: 3s then 1-2 singles
    return sizes, r - 3 * (r // 3)


def partition_d(seq):
    """[(group tuple, is_rs)] plus the trailing singles; rehearsal = singles + last groups up to D_REHEARSE items (>= 1 group)."""
    sizes, ns = rs_sizes(len(seq))
    groups, i = [], 0
    for k in sizes:
        groups.append(tuple(seq[i:i + k]))
        i += k
    singles = list(seq[i:])
    tot, n_reh = len(singles), 0
    for g in reversed(groups):
        if n_reh == 0 or tot + len(g) <= D_REHEARSE:
            tot += len(g)
            n_reh += 1
        else:
            break
    return [(g, j < len(groups) - n_reh) for j, g in enumerate(groups)], singles


def decode_act(dec, k, known, c, x, act):
    mem, L = dec.mem, dec.mem.L
    if known:
        sets = sorted((mem.fam_pos[k].get((i, d), set()) for i, d in known.items()), key=len)
        cand = set(sets[0])
        for s_ in sets[1:]:
            cand &= s_
    else:
        cand = set(mem.fam[k])
    if act is not None:
        cand &= act
    if not cand:
        return None, 0
    if len(cand) == 1:
        return next(iter(cand)), 1
    best, bw = None, 0.0
    if c is not None or x is not None:
        lc = dec.leaves(L.cnt, c) if c is not None else None
        lx = dec.leaves(L.ctx, x) if x is not None else None
        for t in sorted(cand):
            w = 0.0
            for rid in mem.recs_of.get(t, ()):
                e = L.records[rid]
                if e.leaf is None or e.cleaf is None:
                    continue
                if lc is not None and e.cleaf.id not in lc:
                    continue
                if lx is not None and e.leaf.id not in lx:
                    continue
                w += e.weight
            if w > bw:
                best, bw = t, w
    if best is None:
        best = min(cand, key=lambda t: (-mem.span_w.get(t, 0.0), t))
    return best, len(cand)


def recall_rs(dec, groups, frng, f, concepts, act, tally):
    ok_all = True
    for g, rs in groups:
        if not rs:
            continue
        k = len(g)
        c, x = dec.label(g) if concepts else (None, None)
        known = {i: d for i, d in enumerate(g) if frng.random() >= f}
        c = c if (c is not None and frng.random() >= f) else None
        x = x if (x is not None and frng.random() >= f) else None
        tally["n%d" % k] += 1
        if len(known) == k:
            tally["intact%d" % k] += 1
            continue
        got, nc = decode_act(dec, k, known, c, x, act) if k in (3, 4) else (None, 0)
        tally["try%d" % k] += 1
        if got == g:
            tally["resc%d" % k] += 1
        else:
            tally["err%d" % k] += 1
            tally["fail_nocand" if nc == 0 else "fail_wrong"] += 1
            ok_all = False
    return ok_all


def letter_probe_d(mem, rng, f):
    """10-list staircase of consonants from 5: same slots, forgetting and rescue, no storage, no activation."""
    length, hist = 5, []
    dec = Decoder(mem)
    for _ in range(10):
        seq = [rng.choice(CONS) for _ in range(length)]
        groups, _ = partition_d(seq)
        ok = recall_rs(dec, groups, rng, f, False, set(), Counter())
        hist.append((length, int(ok)))
        length = max(1, length + (1 if ok else -1))
    return {"final": length, "max": max(h[0] for h in hist), "hist": hist}


def epam(arm, seed, max_sessions, budget, tag):
    opt = dict(f=0.25, act=True, concepts=True, prior=True)
    opt.update(D_ARMS[arm])
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    mem = Mem(seed)
    info = {"arm": arm, "seed": seed, "rec_w": REC_W, "store_s": STORE_S, "rehearse": D_REHEARSE, "start": 7, **opt}
    if opt["prior"]:
        items, n_pre = prior_stream(seed, 1)
        for c, d in items:
            mem.learn(list(d))
        info.update(n_pre=n_pre, prior_items=len(items), prior_records=len(mem.L.records))
    info["fam_after_prior"] = [len(mem.fam[3]), len(mem.fam[4])]
    print("prior", info, flush=True)
    rng = random.Random(seed * 31 + 5)          # same list stream as P3b / P3c
    prng = random.Random(seed * 77 + 11)
    frng = random.Random(seed * 13 + 7)
    length = 7
    rows, sess_rows, probes = [], [], {}
    stop = "max_sessions"
    for sess in range(1, max_sessions + 1):
        if budget and time.time() - t0 > budget:
            stop = "time_budget"
            break
        act = set()
        sstart, presented = length, []
        for li in range(lists_per_session(sess)):
            digits = [str(rng.randrange(10)) for _ in range(length)]
            groups, singles = partition_d(digits)
            clock, nlearn, nreinf = 0.0, 0, 0
            for g, rs in groups:                 # storage during presentation (singles add time only)
                clock += len(g)
                presented.append(g)
                if mem.span_w.get(g, 0.0) >= REC_W:
                    mem.span_w[g] += 1.0
                    act.add(g)
                    nreinf += 1
                elif clock >= STORE_S:
                    mem.learn(list(g))
                    clock -= STORE_S
                    if mem.span_w.get(g, 0.0) >= REC_W:
                        act.add(g)
                    nlearn += 1
            dec = Decoder(mem)
            tally = Counter()
            ok = recall_rs(dec, groups, frng, opt["f"], opt["concepts"], act if opt["act"] else None, tally)
            row = {"s": sess, "trial": li + 1, "len": length, "ok": int(ok), "learned": nlearn, "reinf": nreinf,
                   "nrs": sum(1 for g in groups if g[1]), "t": round(time.time() - t0, 1)}
            row.update(tally)
            rows.append(row)
            length = max(1, length + (1 if ok else -1))
        miss = sum(1 for g in presented if g not in act)
        sess_rows.append({"s": sess, "start_len": sstart, "end_len": length, "groups": len(presented), "not_stored": miss,
                          "fam3": len(mem.fam[3]), "fam4": len(mem.fam[4])})
        if sess in PROBE_SESSIONS:
            probes[sess] = letter_probe_d(mem, prng, opt["f"])
            print("probe", sess, probes[sess], flush=True)
        if sess % 10 == 0:
            blk = [x["len"] for x in rows if x["s"] > sess - 10]
            print("session %d len(mean last10) %.1f next %d fam3 %d fam4 %d t %.0fs" % (
                sess, sum(blk) / len(blk), length, len(mem.fam[3]), len(mem.fam[4]), time.time() - t0), flush=True)
            epam_dump(tag, arm, seed, info, rows, sess_rows, probes, stop + "(partial)", t0, sess)
    epam_dump(tag, arm, seed, info, rows, sess_rows, probes, stop, t0, sess_rows[-1]["s"] if sess_rows else 0)
    print("done", len(sess_rows), "sessions", round(time.time() - t0), "s", stop, flush=True)


def epam_dump(tag, arm, seed, info, rows, sess_rows, probes, stop, t0, sess):
    d = {"experiment": "psych_digits P3d epam", "cfg": CFG, "info": info, "stopped": stop, "sessions": sess,
         "lists": len(rows), "seconds": round(time.time() - t0, 1), "probes": probes, "sess_rows": sess_rows, "rows": rows}
    fn = os.path.join(OUT, "%s_%s_s%d.json" % (tag, arm, seed))
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    os.replace(fn + ".tmp", fn)


def epamsummary(tag):
    import glob
    runs = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(OUT, "%s_*_s*.json" % tag))):
        d = json.load(open(fn, encoding="utf-8"))
        if d.get("experiment") == "psych_digits P3d epam":
            runs[d["info"]["arm"]][d["info"]["seed"]] = d

    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def ms(vals, fmt="%.1f"):
        v = [x for x in vals if x is not None and x == x]
        return (fmt % (sum(v) / len(v)) + " [" + fmt % min(v) + "-" + fmt % max(v) + "]") if v else "-"
    pts = (15, 50, 100, 250, 500, 865)
    L = ["# P3d: EPAM IV architecture with TRELLIS as the semantic net (generated by psych_digits.py epamsummary)", "",
         "Pre-registration: docs/PLAN_PSYCH.md, section P3d. Span at s = mean list length over the 5-session block ending at s; "
         "mean [min-max] over seeds.", "", "DD: start 7-9; about 80 by session ~500; 104 at ~865.", "", "## Span at sessions", "",
         "| arm | n | " + " | ".join("s%d" % p for p in pts) + " | max block span | first s >= 80 (per seed) |", "|---|---|" + "---|" * (len(pts) + 2)]
    for arm in D_ARMS:
        rr = runs.get(arm, {})
        if not rr:
            continue
        cells = [ms([_span_at(d, p) for d in rr.values() if d["sessions"] >= p]) for p in pts]
        mx = [max((_span_at(d, s) or 0) for s in range(5, d["sessions"] + 1)) for d in rr.values()]
        hits = ", ".join(str(_hit(d, 80) or "-") for _, d in sorted(rr.items()))
        L.append("| %s | %d | %s | %s | %s |" % (arm, len(rr), " | ".join(cells), ms(mx), hits))
    L.append("| DD | - | ~7-9 | - | - | - | ~80 | 104 | 104 | ~500 |")
    L += ["", "## Per-group RS recall (pooled over seeds)", "", "| arm | window | groups | err 3-digit | err 4-digit | intact | rescued of tries | fail: no activated candidate | fail: wrong pick |",
          "|---|---|---|---|---|---|---|---|---|"]
    keys = ["n3", "n4", "err3", "err4", "intact3", "intact4", "try3", "try4", "resc3", "resc4", "fail_nocand", "fail_wrong"]
    for arm in D_ARMS:
        rr = runs.get(arm, {})
        for lo, hi in ((0, 50), (50, 250), (250, 500), (500, 865)):
            c = Counter()
            for d in rr.values():
                c.update(_sum_rows(d, lo, hi, keys))
            n = c["n3"] + c["n4"]
            if not n:
                continue
            tr = c["try3"] + c["try4"]
            L.append("| %s | %d-%d | %d | %s | %s | %.3f | %s | %d | %d |" % (
                arm, lo, hi, n, "%.3f" % (c["err3"] / c["n3"]) if c["n3"] else "-", "%.3f" % (c["err4"] / c["n4"]) if c["n4"] else "-",
                (c["intact3"] + c["intact4"]) / n, "%.3f" % ((c["resc3"] + c["resc4"]) / tr) if tr else "-", c["fail_nocand"], c["fail_wrong"]))
    L += ["", "## Trial order within session (arm full, sessions 250-865, pooled): per-group RS error", ""]
    full = runs.get("full", {})
    to = defaultdict(Counter)
    for d in full.values():
        for r in d["rows"]:
            if 250 < r["s"] <= 865 and r["trial"] <= 6:
                to[r["trial"]].update({"n": r.get("n3", 0) + r.get("n4", 0), "e": r.get("err3", 0) + r.get("err4", 0)})
    L.append("| trial | " + " | ".join(str(t) for t in sorted(to)) + " |")
    L.append("|---|" + "---|" * len(to))
    L.append("| error | " + " | ".join("%.3f (%d)" % (to[t]["e"] / to[t]["n"], to[t]["n"]) if to[t]["n"] else "-" for t in sorted(to)) + " |")
    rel = []
    for d in full.values():
        rs = d["rows"]
        for a, b in zip(rs, rs[1:]):
            if b["trial"] == 1 and 250 < b["s"] <= 865:
                ea, na = a.get("err3", 0) + a.get("err4", 0), a.get("n3", 0) + a.get("n4", 0)
                eb, nb = b.get("err3", 0) + b.get("err4", 0), b.get("n3", 0) + b.get("n4", 0)
                rel.append((ea, na, eb, nb))
    if rel:
        L.append("")
        L.append("Release: last trial of a session %.3f vs first trial of the next %.3f (per-group RS error, pooled)." % (
            sum(r[0] for r in rel) / max(1, sum(r[1] for r in rel)), sum(r[2] for r in rel) / max(1, sum(r[3] for r in rel))))
    L += ["", "## Net size and not-stored share", "", "| arm | familiar 3-digit at end | familiar 4-digit at end | groups not stored, sessions 500-865 |", "|---|---|---|---|"]
    for arm in D_ARMS:
        rr = runs.get(arm, {})
        if not rr:
            continue
        f3 = [d["sess_rows"][-1]["fam3"] for d in rr.values()]
        f4 = [d["sess_rows"][-1]["fam4"] for d in rr.values()]
        ns = [sum(x["not_stored"] for x in d["sess_rows"] if x["s"] > 500) / max(1, sum(x["groups"] for x in d["sess_rows"] if x["s"] > 500)) for d in rr.values()]
        L.append("| %s | %s | %s | %s |" % (arm, ms(f3, "%.0f"), ms(f4, "%.0f"), ms(ns, "%.3f")))
    L.append("| DD (RSS95 estimate) | ~1,000 (substantially all) | a fraction of 10,000 (total ~3,000) | 8% missed in free recall |")
    L += ["", "## Letter probes: final / max length", "", "| arm | " + " | ".join("s%d" % p for p in PROBE_SESSIONS) + " |", "|---|" + "---|" * len(PROBE_SESSIONS)]
    for arm in D_ARMS:
        rr = runs.get(arm, {})
        if rr:
            cells = []
            for p in PROBE_SESSIONS:
                v = [d["probes"][str(p)] for d in rr.values() if str(p) in d["probes"]]
                cells.append("%s / %s" % (ms([x["final"] for x in v]), ms([x["max"] for x in v])) if v else "-")
            L.append("| %s | %s |" % (arm, " | ".join(cells)))
    # pre-registered effects
    L += ["", "## Pre-registered effects (arm full)", ""]
    if full:
        mx = mean([max((_span_at(d, s) or 0) for s in range(5, d["sessions"] + 1)) for d in full.values()])
        L.append("1. Maximum block span >= 20: %.1f -> **%s**." % (mx, "holds" if mx >= 20 else "fails"))
        hs = [_hit(d, 80) for d in full.values()]
        hm = mean(hs) if all(h is not None for h in hs) else None
        verdict = "fails, never" if hm is None else ("holds" if 250 <= hm <= 865 else "fails, too fast")
        L.append("2. First session with span >= 80 in [250, 865]: per seed %s -> **%s**; span at 865 %s (DD 104)." % (
            hs, verdict, ms([_span_at(d, 865) for d in full.values() if d["sessions"] >= 865])))
        c = Counter()
        for d in full.values():
            c.update(_sum_rows(d, 250, 865, keys))
        if c["n3"] and c["n4"]:
            e3, e4 = c["err3"] / c["n3"], c["err4"] / c["n4"]
            L.append("3. 4-digit error > 3-digit error (250-865): %.3f vs %.3f -> **%s** (EPAM 3.0%% vs 1.5%%)." % (e4, e3, "holds" if e4 > e3 else "fails"))
        else:
            L.append("3. 4- vs 3-digit: not reached.")
        po = runs.get("prioroff", {})
        if po:
            a, b = mean([_span_at(d, 50) for d in full.values()]), mean([_span_at(d, 50) for d in po.values()])
            L.append("4. Prior speeds early learning (span s50 full vs prioroff): %.1f vs %.1f -> **%s**." % (a, b, "holds" if a > b else "fails"))
        lp = [d["probes"][str(p)]["final"] for d in full.values() for p in PROBE_SESSIONS if str(p) in d["probes"]]
        L.append("5. Letter probe final <= 9 at every probe: max %s -> **%s**." % (max(lp) if lp else "-", "holds" if lp and max(lp) <= 9 else "fails"))
        for other in ("actoff", "conceptsoff"):
            o = runs.get(other, {})
            if o:
                pairs = [(_span_at(full[s], 865), _span_at(o[s], 865)) for s in full if s in o and full[s]["sessions"] >= 865 and o[s]["sessions"] >= 865]
                L.append("- %s: span s865 full vs %s per seed %s -> full beats it on every seed: **%s**." % (
                    other, other, [("%.1f" % a, "%.1f" % b) for a, b in pairs], all(a > b for a, b in pairs) if pairs else "-"))
    path = os.path.join(OUT, "%s_SUMMARY.md" % tag)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__" and sys.argv[1] == "epam":
    a = sys.argv
    epam(a[2], int(a[3]), int(a[4]) if len(a) > 4 else 865, float(a[5]) if len(a) > 5 else 0.0, a[6] if len(a) > 6 else "P3d")
if __name__ == "__main__" and sys.argv[1] == "epamsummary":
    epamsummary(sys.argv[2] if len(sys.argv) > 2 else "P3d")
