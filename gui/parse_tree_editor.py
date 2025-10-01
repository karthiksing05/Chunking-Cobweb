# editor_api.py
from flask import Flask, jsonify, request
from parse import FiniteParseTree, LanguageChunkingParser
from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
import json

app = Flask(__name__)

# Build a LanguageChunkingParser and produce one FiniteParseTree as a demo:
parser = LanguageChunkingParser(TEST_CORPUS2, context_length=2)
# create a simple LTM by parsing an example window so categorization works
sample_sentence = generate("S", TEST_GRAMMAR2)

## BOOT UP LTM!!
num_load = 50
document = []

for _ in range(num_load):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

for i, doc in enumerate(document):
    parse_tree = parser.parse_input([doc], end_behavior="converge", debug=False)[0]
    parser.add_parse_tree(parse_tree, debug=False)

curr_tree = FiniteParseTree(parser.get_long_term_memory(), parser.id_to_value, parser.value_to_id, context_length=2)
curr_tree.build_primitives(sample_sentence)

# ensure editor state exists
curr_tree._ensure_editor_state()

@app.route("/api/tree", methods=["GET"])
def api_get_tree():
    """
    Return the render-friendly D3 JSON plus pair buttons and action_log
    """
    d3_json = curr_tree._draw_tree_to_json()
    pairs = curr_tree.get_parentless_pairs()
    return jsonify({"tree": d3_json, "pairs": pairs, "action_log": curr_tree.action_log})

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    data = request.get_json()
    left = data.get("left_word_index")
    right = data.get("right_word_index")
    debug = data.get("debug", False)
    try:
        result = curr_tree.evaluate_pair(left, right, debug=debug)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/apply", methods=["POST"])
def api_apply():
    data = request.get_json()
    left = data.get("left_word_index")
    right = data.get("right_word_index")
    try:
        res = curr_tree.apply_candidate(left, right)
        # return updated tree JSON and action_log
        return jsonify({"ok": True, "tree": curr_tree._draw_tree_to_json(), "action_log": curr_tree.action_log, "apply_result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/undo", methods=["POST"])
def api_undo():
    res = curr_tree.undo()
    return jsonify(res if isinstance(res, dict) else {"ok": res, "tree": curr_tree._draw_tree_to_json(), "action_log": curr_tree.action_log})

@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json() or {}
    filepath = data.get("filepath")
    print(filepath)
    if filepath == "":
        filepath = "parse_tree_test.json"
    if not filepath.lower().endswith(".json"):
        filepath += ".json"
    res = curr_tree.export_json(f"tests/gui/parse_tree_editor/{filepath}")
    return jsonify(res)

# optional static UI endpoint to get a rendered HTML (uses FiniteParseTree._build_html)
@app.route("/editor", methods=["GET"])
def editor_page():
    d3_json = json.dumps(curr_tree._draw_tree_to_json())
    html = curr_tree.editor_build_html(d3_json)
    return html

app.run(debug=True, port=5001)
