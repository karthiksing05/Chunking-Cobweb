# hollow_parse_tree_editor_mh.py  –  Flask GUI for manual parse tree annotation
#
# Purely structural: no WEBSTER, no LTM, no Cobweb hierarchies.
# The user picks which adjacent nodes to merge and in what order.
# The only output is a "hollow JSON" recording the sentence and merge order.
#
# Export format:
#   { "sentence": "the dog chased the cat",
#     "merges": [{"left": 0, "right": 1}, {"left": 0.5, "right": 2}, ...] }
#
# These can later be replayed by the unittest to train WEBSTER.

import sys, os

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

from flask import Flask, jsonify, request
from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
import json
import re
import uuid

app = Flask(__name__)

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "hollow_parse_tree_editor_mh")
os.makedirs(EXPORT_DIR, exist_ok=True)

grammar = TEST_GRAMMAR2

# ── Lightweight structural tree (no LTM) ──────────────────────────────────

class HollowNode:
    """A minimal tree node for structural annotation only."""

    def __init__(self, title: str, position_idx: float, children=None):
        self.title = title
        self.position_idx = position_idx
        self.parent = None          # HollowNode | None
        self.children: list = children or []   # sorted by position_idx

    def to_d3(self) -> dict:
        return {
            "title": self.title,
            "children": [ch.to_d3() for ch in
                         sorted(self.children, key=lambda c: c.position_idx)],
        }


class HollowTree:
    """A sentence-level parse tree built entirely by user merge decisions."""

    def __init__(self, sentence: str):
        self.sentence = sentence
        tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
        self.root = HollowNode("ROOT", -1)
        self.leaves: list[HollowNode] = []
        self._undo_stack: list[dict] = []

        for i, tok in enumerate(tokens):
            leaf = HollowNode(tok, i)
            leaf.parent = self.root
            self.root.children.append(leaf)
            self.leaves.append(leaf)

    # — queries ——————————————————————————————————————————————————————

    def get_parentless_pairs(self):
        """Return adjacent pairs that are direct children of root."""
        kids = sorted(self.root.children, key=lambda n: n.position_idx)
        pairs = []
        for i in range(len(kids) - 1):
            l, r = kids[i], kids[i + 1]
            pairs.append([l.title, r.title, l.position_idx, r.position_idx])
        return pairs

    def to_d3(self) -> dict:
        return self.root.to_d3()

    # — mutations —————————————————————————————————————————————————————

    def _find_root_child(self, pos_idx):
        for ch in self.root.children:
            if ch.position_idx == pos_idx:
                return ch
        return None

    def apply_merge(self, left_idx, right_idx):
        left = self._find_root_child(left_idx)
        right = self._find_root_child(right_idx)
        if left is None or right is None:
            raise ValueError(
                f"Cannot find root children at positions {left_idx}, {right_idx}")

        new_pos = 0.5 * (left.position_idx + right.position_idx)
        new_title = f"[{left.title} + {right.title}]"
        merged = HollowNode(new_title, new_pos, children=[left, right])

        # re-parent
        self.root.children.remove(left)
        self.root.children.remove(right)
        left.parent = merged
        right.parent = merged
        merged.parent = self.root
        self.root.children.append(merged)

        self._undo_stack.append({
            "merged": merged,
            "left": left,
            "right": right,
        })
        return {"ok": True, "title": new_title, "position_idx": new_pos}

    def undo(self):
        if not self._undo_stack:
            return {"ok": False, "error": "Nothing to undo"}
        entry = self._undo_stack.pop()
        merged = entry["merged"]
        left = entry["left"]
        right = entry["right"]

        self.root.children.remove(merged)
        left.parent = self.root
        right.parent = self.root
        self.root.children.append(left)
        self.root.children.append(right)
        return {"ok": True}


# ── State ─────────────────────────────────────────────────────────────────
sample_sentence = generate("S", grammar)
tree = HollowTree(sample_sentence)
merge_log: list[dict] = []


def reset_tree(sentence=None):
    global tree, sample_sentence, merge_log
    sample_sentence = sentence or generate("S", grammar)
    tree = HollowTree(sample_sentence)
    merge_log = []
    print(f"[INFO] Sentence loaded: {sample_sentence}")


# ── API endpoints ─────────────────────────────────────────────────────────

@app.route("/api/tree", methods=["GET"])
def api_get_tree():
    return jsonify({
        "tree": tree.to_d3(),
        "pairs": tree.get_parentless_pairs(),
        "merge_log": merge_log,
        "sentence": sample_sentence,
    })


@app.route("/api/set_sentence", methods=["POST"])
def api_set_sentence():
    data = request.get_json() or {}
    sentence = data.get("sentence", "").strip()
    if not sentence:
        sentence = generate("S", grammar)
    reset_tree(sentence)
    return jsonify({"ok": True, "sentence": sample_sentence})


@app.route("/api/apply", methods=["POST"])
def api_apply():
    data = request.get_json()
    left = data.get("left_word_index")
    right = data.get("right_word_index")
    try:
        res = tree.apply_merge(left, right)
        merge_log.append({"left": left, "right": right})
        return jsonify({
            "ok": True,
            "tree": tree.to_d3(),
            "merge_log": merge_log,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/undo", methods=["POST"])
def api_undo():
    tree.undo()
    if merge_log:
        merge_log.pop()
    return jsonify({
        "ok": True,
        "tree": tree.to_d3(),
        "merge_log": merge_log,
    })


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    if not filename:
        filename = "hollow_" + str(uuid.uuid4())[:8]
    if not filename.lower().endswith(".json"):
        filename += ".json"

    hollow = {"sentence": sample_sentence, "merges": list(merge_log)}
    export_path = os.path.join(EXPORT_DIR, filename)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(hollow, f, indent=2)

    reset_tree()
    return jsonify({
        "ok": True,
        "message": f"Exported to {export_path}. New sentence loaded.",
        "exported_path": export_path,
        "new_sentence": sample_sentence,
    })


# ── Editor page ───────────────────────────────────────────────────────────

@app.route("/editor", methods=["GET"])
def editor_page():
    d3_json = json.dumps(tree.to_d3())
    return _build_editor_html(d3_json)


def _build_editor_html(d3_data_json, node_w=200, node_h=40, h_gap=30, v_gap=60):
    sentence_str = sample_sentence or ""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Hollow Parse Tree Editor</title>
<style>
body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
#editor-container {{ display:flex; flex-direction:row; height:calc(100vh - 130px); }}
#tree-panel {{ flex:3; overflow:auto; border-right:1px solid #ccc; padding:12px; }}
#sidebar {{ flex:1; overflow-y:auto; padding:12px; background:#f9f9f9; min-width:250px; }}
#header {{ padding:12px; border-bottom:1px solid #ccc; }}
button {{ margin:4px; padding:4px 8px; font-size:12px; cursor:pointer; }}
#pair-buttons button {{ width:100%; text-align:left; margin:4px 0; padding:6px 10px; background:#e8f0fe; border:1px solid #90b4e0; border-radius:4px; }}
#pair-buttons button:hover {{ background:#d0e0fc; }}
ul {{ list-style:none; padding-left:0; font-size:12px; }}
li {{ margin-bottom:6px; }}
#sentence-input {{ width:60%; padding:4px 8px; font-size:13px; }}
.export-input {{ width:50%; padding:3px 6px; font-size:12px; margin-right:4px; }}
</style>
</head>
<body>
<div id="header">
    <h2 style="margin:0 0 6px 0;">Hollow Parse Tree Editor</h2>
    <div style="margin-bottom:8px;">
        <input id="sentence-input" type="text" placeholder="Type a sentence or leave blank for random…" value="" />
        <button id="load-btn">Load Sentence</button>
    </div>
    <h4 style="margin:0;">Current sentence: <span id="sentence-text">{sentence_str}</span></h4>
    <button id="undo-btn">Undo Last Merge</button>
    <div style="display:inline-block; margin-left:16px;">
        <input id="export-filename" class="export-input" type="text" placeholder="filename (optional)" />
        <button id="export-btn">Export Hollow Tree</button>
    </div>
</div>
<div id="editor-container">
    <div id="tree-panel"><div id="tree"></div></div>
    <div id="sidebar">
        <div id="pair-buttons"><strong>Available Pairs:</strong></div>
        <div style="margin-top:16px;">
            <strong>Merge Log:</strong>
            <ul id="merge-log"></ul>
        </div>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
let treeData = {d3_data_json};
const nodeW={node_w}, nodeH={node_h}, hGap={h_gap}, vGap={v_gap};

/* ─── Tree rendering (reused from parse_tree_editor_mh) ─── */
function renderTree(data) {{
    d3.select("#tree").selectAll("*").remove();
    const root = d3.hierarchy(data);
    const layout = d3.tree().nodeSize([nodeW+hGap, nodeH+vGap])
        .separation((a,b) => a.parent===b.parent ? 1.0 : 1.4);
    layout(root);
    const svg = d3.select("#tree").append("svg").attr("width",1).attr("height",1);
    const g = svg.append("g");

    const linkGen = d3.linkVertical().x(d=>d.x).y(d=>d.y);
    g.selectAll("path.link").data(root.links()).join("path").attr("class","link")
        .attr("fill","none").attr("stroke","#9aa1a9").attr("stroke-width",1.5)
        .attr("d", linkGen);

    const node = g.selectAll("g.node").data(root.descendants()).join("g")
        .attr("transform", d=>`translate(${{d.x}},${{d.y}})`);
    node.append("rect").attr("class","node-box").attr("x",-nodeW/2).attr("y",0)
        .attr("width",nodeW).attr("height",nodeH)
        .attr("stroke","#444").attr("fill","#fff").attr("rx",8).attr("ry",8);

    node.each(function(d) {{
        const g2 = d3.select(this);
        g2.append("text").attr("x", 0).attr("y", nodeH/2 + 4)
          .attr("text-anchor","middle").attr("font-size","12px")
          .attr("fill","#333").text(d.data.title || "");
    }});

    // fit viewBox
    const bbox = g.node().getBBox();
    svg.attr("viewBox", `${{bbox.x-20}} ${{bbox.y-20}} ${{bbox.width+40}} ${{bbox.height+40}}`)
       .attr("width", bbox.width+40).attr("height", bbox.height+40);
}}

/* ─── Refresh UI from server ─── */
function refresh() {{
    fetch("/api/tree").then(r=>r.json()).then(data => {{
        treeData = data.tree;
        renderTree(treeData);
        document.getElementById("sentence-text").innerText = data.sentence;

        // pair buttons
        const pb = document.getElementById("pair-buttons");
        pb.innerHTML = "<strong>Available Pairs:</strong>";
        (data.pairs || []).forEach(p => {{
            const btn = document.createElement("button");
            btn.textContent = p[0] + "  +  " + p[1];
            btn.onclick = () => applyMerge(p[2], p[3]);
            pb.appendChild(btn);
        }});

        // merge log
        const ml = document.getElementById("merge-log");
        ml.innerHTML = "";
        (data.merge_log || []).forEach((m, i) => {{
            const li = document.createElement("li");
            li.textContent = (i+1) + ". merge(" + m.left + ", " + m.right + ")";
            ml.appendChild(li);
        }});
    }});
}}

/* ─── Actions ─── */
function applyMerge(left, right) {{
    fetch("/api/apply", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{left_word_index: left, right_word_index: right}})
    }}).then(r => r.json()).then(data => {{
        if (data.ok) refresh();
        else alert("Merge failed: " + (data.error || "unknown error"));
    }});
}}

document.getElementById("undo-btn").onclick = () => {{
    fetch("/api/undo", {{method:"POST"}}).then(r=>r.json()).then(() => refresh());
}};

document.getElementById("load-btn").onclick = () => {{
    const sentence = document.getElementById("sentence-input").value.trim();
    fetch("/api/set_sentence", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{sentence}})
    }}).then(r => r.json()).then(data => {{
        if (data.ok) {{
            document.getElementById("sentence-input").value = "";
            refresh();
        }}
    }});
}};

document.getElementById("export-btn").onclick = () => {{
    const filename = document.getElementById("export-filename").value.trim();
    fetch("/api/export", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{filename}})
    }}).then(r => r.json()).then(data => {{
        if (data.ok) {{
            alert(data.message);
            document.getElementById("export-filename").value = "";
            refresh();
        }}
    }});
}};

/* ─── Initial render ─── */
refresh();
</script>
</body>
</html>"""


app.run(debug=True, port=5002)
