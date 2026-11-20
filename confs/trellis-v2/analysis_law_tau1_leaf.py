runs = {"1,000 s13": dict(leaf=(.558,.206,.235), lv={"b7":(.331,.255),"b6":(.197,.208),"b4":(.107,.140),"root":(.034,.095)}),
        "4,000 s13": dict(leaf=(.580,.193,.228), lv={"b7":(.293,.211),"b6":(.180,.161),"b4":(.087,.116),"root":(.029,.082)}),
        "4,000 s17": dict(leaf=(.541,.218,.241), lv={"b7":(.323,.265),"b6":(.201,.225),"b4":(.116,.157),"root":(.033,.083)})}
nul = {"1,000 s13":(.167,.078),"4,000 s13":(.122,.099),"4,000 s17":(.113,.086)}
for k, r in runs.items():
    s, y, n = r["leaf"]; ls = s + y
    print(k, "leaf syn pass %.3f meaning %.3f share %.3f" % (ls, s, y / ls))
    for l, (a, b) in r["lv"].items():
        print("  ", l, "syn ret %.2f mean ret %.2f share %.3f" % ((a + b) / ls, a / s, b / (a + b)))
    a, b = nul[k]; print("   nocls share %.3f syn-pass-of-input %.3f" % (b / (a + b), a + b))
