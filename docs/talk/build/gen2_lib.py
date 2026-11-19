# helpers appended to Slide: trees, boards, chips
RED_T = "#F6DAD5"
CHUNK_COLS = ["#F4C7C3", "#FCE8B2", "#B7E1CD", "#C6DAFC", "#D9C8F0", "#FAD7B5", "#BFE6E6"]


def _height(t):
    return 0 if isinstance(t, str) else 1 + max(_height(c) for c in t)


def tree(self, k, t, x0, y0, sp, dy, size=14, boxw=0.95, dot=0.13, fill=BLUE):
    """Draw nested-list binary tree t. Leaves are strings. Leaves spaced sp apart starting x0 (left edge of first leaf cell)."""
    H = _height(t)
    cnt = [0]
    nodes = []  # (id, x, y, is_leaf)
    lines = []

    def walk(n, path):
        if isinstance(n, str):
            x = x0 + sp * (cnt[0] + 0.5)
            cnt[0] += 1
            y = y0 + H * dy
            self.text(f"{k}w{len(nodes)}", x - boxw / 2, y - 0.02, boxw, 0.4, n, size=size, align="CENTER")
            nodes.append((x, y, True))
            return x, y, True
        kids = [walk(c, path + "x") for c in n]
        x = sum(c[0] for c in kids) / len(kids)
        y = y0 + (H - _height(n)) * dy
        for c in kids:
            lines.append((x, y, c[0], c[1], c[2]))
        e = {"id": f"{self.p}_{k}d{len(nodes)}", "type": "ellipse", "x": x - dot / 2, "y": y - dot / 2, "w": dot, "h": dot, "fill": fill}
        self.els.append(e)
        nodes.append((x, y, False))
        return x, y, False

    walk(t, "")
    for i, (px, py, cx, cy, leaf) in enumerate(lines):
        import math
        dx, dyy = cx - px, cy - py
        L = math.hypot(dx, dyy) or 1
        a = dot / 2 + 0.01
        b = (0.03 if leaf else dot / 2 + 0.01)
        self.line(f"{k}l{i}", px + dx / L * a, py + dyy / L * a, cx - dx / L * b, cy - dyy / L * b, color="#8A96A3", wt=1.25, plain=True)
    return x0 + sp * cnt[0], y0 + H * dy + 0.4


def chip(self, k, x, y, w, h, text, ok, size=12):
    return self.box(k, x, y, w, h, text, GREEN_T if ok else RED_T, size=size, bold=False)


def board(self, k, x, y, cw, rh, pieces, chunks, flagged=(), size=12):
    """9x9 table: file/rank labels + squares. pieces {sq: 'wK'}; chunks list of lists of squares; flagged squares marked red."""
    oid = f"{self.p}_{k}"
    ex = self.extras
    ex.append({"createTable": {"objectId": oid, "elementProperties": {"pageObjectId": self.id,
               "size": {"width": {"magnitude": E(cw * 9), "unit": "EMU"}, "height": {"magnitude": E(rh * 9), "unit": "EMU"}},
               "transform": {"scaleX": 1, "scaleY": 1, "translateX": E(x), "translateY": E(y), "unit": "EMU"}}, "rows": 9, "columns": 9}})
    ex.append({"updateTableColumnProperties": {"objectId": oid, "tableColumnProperties": {"columnWidth": {"magnitude": E(cw), "unit": "EMU"}}, "fields": "columnWidth"}})
    ex.append({"updateTableRowProperties": {"objectId": oid, "tableRowProperties": {"minRowHeight": {"magnitude": E(rh), "unit": "EMU"}}, "fields": "minRowHeight"}})
    cmap = {}
    for i, ch in enumerate(chunks):
        for sq in ch:
            cmap[sq] = CHUNK_COLS[i % len(CHUNK_COLS)]
    def cell_fill(r, c, col):
        ex.append({"updateTableCellProperties": {"objectId": oid, "tableRange": {"location": {"rowIndex": r, "columnIndex": c}, "rowSpan": 1, "columnSpan": 1},
                   "tableCellProperties": {"tableCellBackgroundFill": solid(col), "contentAlignment": "MIDDLE"}, "fields": "tableCellBackgroundFill.solidFill.color,contentAlignment"}})
    def cell_text(r, c, txt, bold, color, sz):
        loc = {"rowIndex": r, "columnIndex": c}
        ex.append({"insertText": {"objectId": oid, "cellLocation": loc, "text": txt}})
        ex.append({"updateTextStyle": {"objectId": oid, "cellLocation": loc, "textRange": {"type": "ALL"}, "style": {"fontSize": {"magnitude": sz, "unit": "PT"},
                   "fontFamily": BF, "bold": bold, "foregroundColor": {"opaqueColor": {"rgbColor": rgb(color)}}}, "fields": "fontSize,fontFamily,bold,foregroundColor"}})
        ex.append({"updateParagraphStyle": {"objectId": oid, "cellLocation": loc, "textRange": {"type": "ALL"}, "style": {"alignment": "CENTER"}, "fields": "alignment"}})
    ex.append({"updateTableBorderProperties": {"objectId": oid, "borderPosition": "ALL", "tableBorderProperties": {"tableBorderFill": solid("#FFFFFF"), "weight": {"magnitude": 1, "unit": "PT"}, "dashStyle": "SOLID"}, "fields": "tableBorderFill,weight,dashStyle"}})
    ex.append({"updateTableCellProperties": {"objectId": oid, "tableCellProperties": {"tableCellBackgroundFill": solid("#EAEEF2"), "contentAlignment": "MIDDLE"}, "fields": "tableCellBackgroundFill.solidFill.color,contentAlignment"}})
    ex.append({"updateTableCellProperties": {"objectId": oid, "tableRange": {"location": {"rowIndex": 8, "columnIndex": 0}, "rowSpan": 1, "columnSpan": 9}, "tableCellProperties": {"tableCellBackgroundFill": solid(BG)}, "fields": "tableCellBackgroundFill.solidFill.color"}})
    ex.append({"updateTableCellProperties": {"objectId": oid, "tableRange": {"location": {"rowIndex": 0, "columnIndex": 0}, "rowSpan": 9, "columnSpan": 1}, "tableCellProperties": {"tableCellBackgroundFill": solid(BG)}, "fields": "tableCellBackgroundFill.solidFill.color"}})
    for c in range(1, 9):
        cell_text(8, c, "abcdefgh"[c - 1], False, GRAY, 11)
    for r in range(8):
        cell_text(r, 0, str(8 - r), False, GRAY, 11)
    for sq, col in cmap.items():
        cell_fill(8 - int(sq[1]), "abcdefgh".index(sq[0]) + 1, col)
    for sq in flagged:
        cell_fill(8 - int(sq[1]), "abcdefgh".index(sq[0]) + 1, RED_T)
    for sq, pc in pieces.items():
        r, c = 8 - int(sq[1]), "abcdefgh".index(sq[0]) + 1
        if sq in flagged:
            cell_text(r, c, "?", True, "#8A2B20", size)
        else:
            cell_text(r, c, pc[1].upper() if pc[0] == "w" else pc[1].lower(), pc[0] == "w", INK, size)
    return oid


Slide.tree = tree
Slide.chip = chip
Slide.board = board
