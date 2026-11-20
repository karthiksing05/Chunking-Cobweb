"""Compact batches: duplicate slide 16 as a template for each figure slide, then swap text and add the image.
Reads the D list from gen3.py. Writes fig_b1.json .. fig_b3.json and fig_renum.json."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "gen3.py")).read()
ns = {"__file__": os.path.join(HERE, "gen3.py")}
exec(src.split("# new slide index order")[0].split("D = [", 1)[0] + "D = [" + src.split("D = [", 1)[1].split("# new slide index order")[0], ns)
D = ns["D"]; U = ns["U"]; GREY = ns["GREY"]
order = []
for old in range(1, 30):
    order.append(("old", old))
    for d in D:
        if d["after"] == old: order.append(("new", d["id"]))
newidx = {v: i for i, (k, v) in enumerate(order) if k == "new"}
oldidx = {v: i for i, (k, v) in enumerate(order) if k == "old"}
def settext(oid, text):
    return [{"deleteText": {"objectId": oid, "textRange": {"type": "ALL"}}}, {"insertText": {"objectId": oid, "insertionIndex": 0, "text": text}}]
def sty(oid, a, b, sz, bold=False, grey=False):
    return {"updateTextStyle": {"objectId": oid, "textRange": {"type": "FIXED_RANGE", "startIndex": a, "endIndex": b},
            "style": {"fontSize": {"magnitude": sz, "unit": "PT"}, "bold": bold, "foregroundColor": {"opaqueColor": ({"rgbColor": GREY} if grey else {"themeColor": "DARK2"})}},
            "fields": "fontSize,bold,foregroundColor"}}
slides = []
for d in D:
    i = d["id"]; s = i + "_slide"; idx = newidx[i]
    r = [{"duplicateObject": {"objectId": "t16_slide", "objectIds": {"t16_slide": s, "t16_num": i + "_num", "t16_title": i + "_title", "t16_op": i + "_op", "t16_tbl": i + "_tbl", "t16_ft": i + "_ft"}}},
         {"deleteObject": {"objectId": i + "_tbl"}}]
    r += settext(i + "_title", d["title"]) + settext(i + "_num", str(idx + 1))
    pw, ph = d["px"]; ar = pw / ph
    w = min(d["maxw"], d["maxh"] * ar); h = w / ar
    def img(x, y):
        return {"createImage": {"objectId": i + "_img", "url": U + d["img"], "elementProperties": {"pageObjectId": s,
                "size": {"width": {"magnitude": int(w), "unit": "EMU"}, "height": {"magnitude": int(h), "unit": "EMU"}},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": int(x), "translateY": y, "unit": "EMU"}}}}
    if d["kind"] == "W":
        r += settext(i + "_op", d["lead"]) + settext(i + "_ft", d["src"])
        r.append(img(457200 + (8229600 - w) / 2, 1290000))
        r += [{"createShape": {"objectId": i + "_take", "shapeType": "TEXT_BOX", "elementProperties": {"pageObjectId": s,
               "size": {"width": {"magnitude": 8229600, "unit": "EMU"}, "height": {"magnitude": 520000, "unit": "EMU"}},
               "transform": {"scaleX": 1, "scaleY": 1, "translateX": 457200, "translateY": 3790000, "unit": "EMU"}}}},
              {"insertText": {"objectId": i + "_take", "text": d["take"]}},
              {"updateTextStyle": {"objectId": i + "_take", "style": {"fontFamily": "Lato", "fontSize": {"magnitude": 16, "unit": "PT"}, "bold": True,
               "foregroundColor": {"opaqueColor": {"themeColor": "DARK2"}}}, "fields": "fontFamily,fontSize,bold,foregroundColor"}}]
    else:
        tx = int(457200 + w + 320040); tw = 8686800 - tx
        t = d["lead"] + "\n" + d["take"] + "\n" + d["src"]
        a, b = len(d["lead"]) + 1, len(d["lead"]) + 1 + len(d["take"]) + 1
        r += [{"deleteObject": {"objectId": i + "_ft"}}] + settext(i + "_op", t)
        r.append({"updatePageElementTransform": {"objectId": i + "_op", "applyMode": "ABSOLUTE", "transform": {"scaleX": round(tw / 3000000, 4), "scaleY": round(3500000 / 3000000, 4), "translateX": tx, "translateY": 950000, "unit": "EMU"}}})
        r += [sty(i + "_op", a, b - 1, 16, True), sty(i + "_op", b, len(t), 12, False, True),
              {"updateParagraphStyle": {"objectId": i + "_op", "textRange": {"type": "ALL"}, "style": {"spaceBelow": {"magnitude": 12, "unit": "PT"}}, "fields": "spaceBelow"}}]
        r.append(img(457200, 950000))
    slides.append(r)
json.dump(slides, open(os.path.join(HERE, "fig_slides.json"), "w"))
json.dump({"newidx": newidx, "oldidx": oldidx}, open(os.path.join(HERE, "fig_idx.json"), "w"))
ren = []
for old in range(1, 30):
    n = oldidx[old] + 1
    if n != old:
        ren += settext("t%02d_num" % old, str(n))
json.dump(ren, open(os.path.join(HERE, "fig_renum.json"), "w"))
print(len(json.dumps(slides)), len(ren))
