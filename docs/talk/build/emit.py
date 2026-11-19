EMU = 914400


def r3(v):
    return round(v, 3)


def rgb(h):
    h = h.lstrip("#")
    return {"red": r3(int(h[0:2], 16) / 255), "green": r3(int(h[2:4], 16) / 255), "blue": r3(int(h[4:6], 16) / 255)}


def E(v):
    return int(round(v * EMU))


def props(pid, x, y, w, h):
    return {"pageObjectId": pid, "size": {"width": {"magnitude": E(w), "unit": "EMU"}, "height": {"magnitude": E(h), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": 1, "translateX": E(x), "translateY": E(y), "unit": "EMU"}}


def tstyle(oid, el, bold, cell=None):
    st = {"fontSize": {"magnitude": el["size"], "unit": "PT"}, "fontFamily": el["font"]}
    f = "fontSize,fontFamily"
    if el["color"] != "#1F2933":
        st["foregroundColor"] = {"opaqueColor": {"rgbColor": rgb(el["color"])}}
        f += ",foregroundColor"
    if bold:
        st["bold"] = True
        f += ",bold"
    r = {"objectId": oid, "textRange": {"type": "ALL"}, "style": st, "fields": f}
    if cell:
        r["cellLocation"] = {"rowIndex": cell[0], "columnIndex": cell[1]}
    return {"updateTextStyle": r}


SH = {"text": "TEXT_BOX", "rect": "RECTANGLE", "round_rect": "ROUND_RECTANGLE", "ellipse": "ELLIPSE"}


def emit_slide(s):
    reqs = [{"createSlide": {"objectId": s.id, "slideLayoutReference": {"predefinedLayout": "BLANK"}}}]
    for el in s.els:
        oid = el["id"]
        if el["type"] == "table":
            rows, widths = el["rows"], el["col_widths"]
            nr, nc = len(rows), len(widths)
            rh = el["size"] * 1.2 / 72 + 0.2
            reqs.append({"createTable": {"objectId": oid, "elementProperties": props(s.id, el["x"], el["y"], sum(widths), rh * nr), "rows": nr, "columns": nc}})
            for c, w in enumerate(widths):
                reqs.append({"updateTableColumnProperties": {"objectId": oid, "columnIndices": [c], "tableColumnProperties": {"columnWidth": {"magnitude": E(w), "unit": "EMU"}}, "fields": "columnWidth"}})
            al = el.get("col_align")
            for r in range(nr):
                for c in range(nc):
                    v = rows[r][c]
                    if not v:
                        continue
                    loc = {"rowIndex": r, "columnIndex": c}
                    reqs.append({"insertText": {"objectId": oid, "cellLocation": loc, "text": v}})
                    reqs.append(tstyle(oid, el, r == 0 or c == 0, (r, c)))
                    if al and al[c] != "START":
                        reqs.append({"updateParagraphStyle": {"objectId": oid, "cellLocation": loc, "textRange": {"type": "ALL"}, "style": {"alignment": al[c]}, "fields": "alignment"}})
            continue
        reqs.append({"createShape": {"objectId": oid, "shapeType": SH[el["type"]], "elementProperties": props(s.id, el["x"], el["y"], el["w"], el["h"])}})
        if "fill" in el:
            reqs.append({"updateShapeProperties": {"objectId": oid, "shapeProperties": {"shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": rgb(el["fill"])}}}, "outline": {"propertyState": "NOT_RENDERED"}}, "fields": "shapeBackgroundFill.solidFill.color,outline.propertyState"}})
        if el.get("text"):
            reqs.append({"insertText": {"objectId": oid, "text": el["text"]}})
            reqs.append(tstyle(oid, el, el.get("bold")))
            para = {}
            if "align" in el:
                para["alignment"] = el["align"]
            if "space_below" in el:
                para["spaceBelow"] = {"magnitude": el["space_below"], "unit": "PT"}
            if para:
                reqs.append({"updateParagraphStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": para, "fields": ",".join(para)}})
            if el.get("bullets"):
                reqs.append({"createParagraphBullets": {"objectId": oid, "textRange": {"type": "ALL"}, "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"}})
    return reqs
