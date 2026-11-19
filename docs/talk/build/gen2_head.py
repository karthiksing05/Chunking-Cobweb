#!/usr/bin/env python3
"""Generate update_presentation batches for the TRELLIS v2 talk. Usage: gen.py GROUP REVISION"""
import json, subprocess, sys, os

HELPER = "/nethome/ksingara3/.claude/skills/synced/e8dbf770-ea33-44f5-bb64-4740d9e5a1a8_20a31c0c-42bf-4633-a895-06ceaf28d34d/google-workspace/scripts/slides_helper.py"
OUT = "/coc/flash10/ksingara3/Chunking-Cobweb/docs/talk/build"
BLUE, GREEN, INK, GRAY, BG = "#3B6E8F", "#5E8C6A", "#1F2933", "#6B7280", "#FAFAFA"
BLUE_T, GREEN_T, NEUT_T, COL_T = "#DCE8F0", "#DFEBE3", "#E8ECEF", "#EEF3F7"
BORDER, HDRB = "#E0E0E0", "#9CA3AF"
HF, BF, MF = "Merriweather", "Lato", "Roboto Mono"
EMU = 914400


from emit import rgb, emit_slide, E


def solid(h):
    return {"solidFill": {"color": {"rgbColor": rgb(h)}}}


class Slide:
    def __init__(self, n, dark=False):
        self.n, self.dark = n, dark
        self.id = f"t{n:02d}_slide"
        self.p = f"t{n:02d}"
        self.els, self.extras = [], []
        self.extras.append({"updatePageProperties": {"objectId": self.id, "pageProperties": {
            "pageBackgroundFill": solid("#22303C" if dark else BG)}, "fields": "pageBackgroundFill.solidFill.color"}})
        self.text("num", 9.0, 5.1, 0.5, 0.4, str(n), size=12, color="#AEB8C2" if dark else GRAY, align="END")

    def text(self, k, x, y, w, h, text, size=16, bold=False, color=INK, font=BF, align=None, bullets=False, sb=None):
        e = {"id": f"{self.p}_{k}", "type": "text", "x": x, "y": y, "w": w, "h": h, "text": text, "size": size,
             "bold": bold, "color": color, "font": font}
        if align: e["align"] = align
        if bullets: e["bullets"] = True
        if sb is not None: e["space_below"] = sb
        self.els.append(e)
        return e["id"]

    def title(self, text, size=24, h=1.0, color=INK):
        return self.text("title", 0.5, 0.3, 9.0, h, text, size=size, bold=True, color=color, font=HF)

    def foot(self, text, y=4.8, h=0.45):
        return self.text("foot", 0.5, y, 8.3, h, text, size=12, color=GRAY)

    def box(self, k, x, y, w, h, text, fill, size=16, bold=False, lead=False, font=BF, shape="round_rect", color=INK, align="CENTER"):
        e = {"id": f"{self.p}_{k}", "type": shape, "x": x, "y": y, "w": w, "h": h, "fill": fill, "text": text,
             "size": size, "bold": bold, "color": color, "font": font, "align": align}
        self.els.append(e)
        if lead:
            n = len(text.split("\n")[0])
            self.style(e["id"], 0, n, {"bold": True}, "bold")
        return e["id"]

    def table(self, k, x, y, w, rows, widths, size=14, aligns=None, tint=None, tint_col=None, hdr_colors=None, mono=None, h=None):
        oid = f"{self.p}_{k}"
        e = {"id": oid, "type": "table", "x": x, "y": y, "w": w, "rows": rows, "header": True, "size": size,
             "col_widths": widths, "font": BF, "color": INK}
        if aligns: e["col_align"] = aligns
        self.els.append(e)
        nr, nc = len(rows), len(widths)
        for (c, col) in (hdr_colors or {}).items():
            self.style(oid, 0, len(rows[0][c]), {"foregroundColor": {"opaqueColor": {"rgbColor": rgb(col)}}}, "foregroundColor", cell=(0, c))
        for (r, c, sub) in (mono or []):
            i = rows[r][c].find(sub)
            self.style(oid, i, i + len(sub), {"fontFamily": MF}, "fontFamily", cell=(r, c))
        def tint_req(r0, c0, rs, cs, col):
            self.extras.append({"updateTableCellProperties": {"objectId": oid, "tableRange": {
                "location": {"rowIndex": r0, "columnIndex": c0}, "rowSpan": rs, "columnSpan": cs},
                "tableCellProperties": {"tableCellBackgroundFill": solid(col)},
                "fields": "tableCellBackgroundFill.solidFill.color"}})
        if tint is not None: tint_req(tint, 0, 1, nc, COL_T)
        if tint_col is not None: tint_req(1, tint_col, nr - 1, 1, COL_T)
        def border(pos, col, rng=None):
            r = {"objectId": oid, "borderPosition": pos, "tableBorderProperties": {
                "tableBorderFill": solid(col), "weight": {"magnitude": 1, "unit": "PT"}, "dashStyle": "SOLID"},
                "fields": "tableBorderFill,weight,dashStyle"}
            if rng: r["tableRange"] = rng
            self.extras.append({"updateTableBorderProperties": r})
        border("ALL", BORDER)
        for pos in ("INNER_VERTICAL",):
            border(pos, BG)
        border("BOTTOM", HDRB, {"location": {"rowIndex": 0, "columnIndex": 0}, "rowSpan": 1, "columnSpan": nc})
        return oid

    def style(self, oid, a, b, style, fields, cell=None):
        r = {"objectId": oid, "textRange": {"type": "FIXED_RANGE", "startIndex": a, "endIndex": b}, "style": style, "fields": fields}
        if cell: r["cellLocation"] = {"rowIndex": cell[0], "columnIndex": cell[1]}
        self.extras.append({"updateTextStyle": r})

    def mono(self, oid, text, sub):
        i = text.find(sub)
        self.style(oid, i, i + len(sub), {"fontFamily": MF}, "fontFamily")

    def bold_sub(self, oid, text, sub):
        i = text.find(sub)
        self.style(oid, i, i + len(sub), {"bold": True}, "bold")

    def line(self, k, x1, y1, x2, y2, arrow=False, color=GRAY, wt=1.5, plain=False):
        x, y, W, H = min(x1, x2), min(y1, y2), max(abs(x2 - x1), 0.003), abs(y2 - y1)
        flip = (x2 - x1) * (y2 - y1) < 0
        oid = f"{self.p}_{k}"
        self.extras.append({"createLine": {"objectId": oid, "lineCategory": "STRAIGHT", "elementProperties": {
            "pageObjectId": self.id,
            "size": {"width": {"magnitude": int(W * EMU), "unit": "EMU"}, "height": {"magnitude": int(H * EMU), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": -1 if flip else 1, "translateX": int(x * EMU),
                          "translateY": int((y + H if flip else y) * EMU), "unit": "EMU"}}}})
        if plain:
            return oid
        lp = {"lineFill": solid(color), "weight": {"magnitude": wt, "unit": "PT"}}
        f = "lineFill,weight"
        if arrow:
            if x2 < x1: lp["startArrow"] = "FILL_ARROW"; f += ",startArrow"
            else: lp["endArrow"] = "FILL_ARROW"; f += ",endArrow"
        self.extras.append({"updateLineProperties": {"objectId": oid, "lineProperties": lp, "fields": f}})

    def spec(self):
        return {"id": self.id, "layout": "BLANK", "elements": self.els}


