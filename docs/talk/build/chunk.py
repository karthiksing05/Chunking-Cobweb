import json, sys, os
sys.argv = ['x']
exec(open('/coc/flash10/ksingara3/Chunking-Cobweb/docs/talk/build/gen2.py').read().split('if __name__')[0])
LIM = 25000
allreq = []
for n in range(1, 30):
    allreq.extend(emit_slide(slides[n]) + slides[n].extras)
chunks, cur, sz = [], [], 0
for r in allreq:
    l = len(json.dumps(r, separators=(',', ':'), ensure_ascii=False)) + 1
    if sz + l > LIM and cur:
        chunks.append(cur); cur, sz = [], 0
    cur.append(r); sz += l
chunks.append(cur)
for i, c in enumerate(chunks):
    json.dump(c, open(f'/coc/flash10/ksingara3/Chunking-Cobweb/docs/talk/build/c_{i:02d}.json', 'w', encoding='utf-8'), separators=(',', ':'), ensure_ascii=False)
    print(i, len(c), len(json.dumps(c, separators=(',', ':'), ensure_ascii=False)))
json.dump({str(k): v for k, v in NOTES.items()}, open('/coc/flash10/ksingara3/Chunking-Cobweb/docs/talk/build/notes2.json', 'w', encoding='utf-8'), ensure_ascii=False)
