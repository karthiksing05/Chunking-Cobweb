"""Position sets for the place-transfer test: do chunks learned on some squares
help code the same shapes on other squares?

Learn set: move-30 positions from ECO C games (1.e4 e5 and the French) whose white
king stands on files e-h. Test sets, 1,000 positions each, none of them learned:
  control  more such positions (the same distribution as the learn set)
  family   positions from ECO D and E games (queen's-pawn openings)
  shift    unlearned positions of the control's kind moved one file, away from the edge
           file with fewer pieces, whose pieces fall off (at most 3, both kings kept):
           shapes clear of that file keep their form, every square changes (not legal
           chess, a probe of square-bound chunks)
  dropped  the same positions with the fallen pieces removed but nothing moved (the
           shift's matched control: it separates moving from losing pieces)
  wing    ECO C positions whose white king stands on files a-c (castled queenside),
           held out of learning

Usage: chess_transfer_sets.py <seed> <out dir>
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
seed, out = int(sys.argv[1]), sys.argv[2]


def rows(fen):
    """Board FEN -> 8 strings of 8 squares ('.' empty), rank 8 first."""
    out = []
    for r in fen.split("/"):
        s = ""
        for ch in r:
            s += "." * int(ch) if ch.isdigit() else ch
        out.append(s)
    return out


def fen_of(rs):
    out = []
    for s in rs:
        r, n = "", 0
        for ch in s:
            if ch == ".":
                n += 1
            else:
                r += (str(n) if n else "") + ch
                n = 0
        out.append(r + (str(n) if n else ""))
    return "/".join(out)


def white_king_file(fen):
    for s in rows(fen):
        if "K" in s:
            return s.index("K")


def shift(fen):
    """(shifted, dropped): the position moved one file, and the same position with
    only the fallen pieces removed (its matched control)."""
    rs = rows(fen)
    lost_h, lost_a = [s[7] for s in rs if s[7] != "."], [s[0] for s in rs if s[0] != "."]
    if len(lost_h) <= len(lost_a):
        lost, moved, kept = lost_h, ["." + s[:7] for s in rs], [s[:7] + "." for s in rs]
    else:
        lost, moved, kept = lost_a, [s[1:] + "." for s in rs], ["." + s[1:] for s in rs]
    if len(lost) > 3 or "K" in lost or "k" in lost:
        return None
    return fen_of(moved), fen_of(kept)


games = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "data", "chess", "settings", "games.tsv"), encoding="utf-8")][1:]
pos = [(g[1], g[6]) for g in games if len(g) > 6 and g[6]]
rng = random.Random(seed)
rng.shuffle(pos)
c_east = [f for eco, f in pos if eco.startswith("C") and white_king_file(f) >= 4]
c_west = [f for eco, f in pos if eco.startswith("C") and white_king_file(f) <= 2]
de = [f for eco, f in pos if eco[:1] in ("D", "E")]
learn, control = c_east[:4000], c_east[4000:5000]
pairs = [p for p in (shift(f) for f in c_east[4000:]) if p][:1000]
sets = {"learn": learn, "control": control, "family": de[:1000], "shift": [m for m, _ in pairs],
        "dropped": [k for _, k in pairs], "wing": c_west[:1000]}
os.makedirs(out, exist_ok=True)
for name, fens in sets.items():
    with open(os.path.join(out, f"{name}.fen"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(fens) + "\n")
    print(name, len(fens), flush=True)
