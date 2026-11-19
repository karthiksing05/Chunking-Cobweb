"""Round 218: the grid of parsing cells. name -> dict(kw=Trellis3R218 knobs, search=None|"beam", learn=bool (win_learn), beam=int).
win_W counts symbols (tokens or frozen chunks); stride defaults to W; D = levels of chunks frozen per IO run. n=1000 stories, sentences are 3..8 tokens,
so W >= 8 is full IO (not listed)."""
CELLS = {
    "full":    dict(kw={}, group="eval"),
    "greedy":  dict(kw={}, search="beam", beam=1, group="eval"),      # v1-style: join the best pair, repeat (Trellis3 search="beam", beam=1)
    "W2":      dict(kw=dict(win_W=2), group="eval"),
    "W3":      dict(kw=dict(win_W=3), group="eval"),
    "W4":      dict(kw=dict(win_W=4), group="eval"),
    "W6":      dict(kw=dict(win_W=6), group="eval"),
    "W3s2":    dict(kw=dict(win_W=3, win_stride=2), group="eval"),
    "W4s2":    dict(kw=dict(win_W=4, win_stride=2), group="eval"),
    "W6s3":    dict(kw=dict(win_W=6, win_stride=3), group="eval"),
    "D1":      dict(kw=dict(io_depth_D=1), group="eval"),
    "D2":      dict(kw=dict(io_depth_D=2), group="eval"),
    "D3":      dict(kw=dict(io_depth_D=3), group="eval"),
    "W4D2":    dict(kw=dict(win_W=4, io_depth_D=2), group="eval"),
    "D1c":     dict(kw=dict(io_depth_D=1, io_depth_mode="chart"), group="eval"),     # chart capped at 2 symbols per run
    "D2c":     dict(kw=dict(io_depth_D=2, io_depth_mode="chart"), group="eval"),     # at 4 symbols
    "D3c":     dict(kw=dict(io_depth_D=3, io_depth_mode="chart"), group="eval"),     # at 8 symbols = full IO for these sentences
    "W6D2c":   dict(kw=dict(win_W=6, io_depth_D=2, io_depth_mode="chart"), group="eval"),
    "W3L":     dict(kw=dict(win_W=3, win_learn=True), group="learn"),
    "W4L":     dict(kw=dict(win_W=4, win_learn=True), group="learn"),
    "D1L":     dict(kw=dict(io_depth_D=1, win_learn=True), group="learn"),
    "D2L":     dict(kw=dict(io_depth_D=2, win_learn=True), group="learn"),
    "D3L":     dict(kw=dict(io_depth_D=3, win_learn=True), group="learn"),
    "W4D2L":   dict(kw=dict(win_W=4, io_depth_D=2, win_learn=True), group="learn"),
    "D1cL":    dict(kw=dict(io_depth_D=1, io_depth_mode="chart", win_learn=True), group="learn"),
    "D2cL":    dict(kw=dict(io_depth_D=2, io_depth_mode="chart", win_learn=True), group="learn"),
}
