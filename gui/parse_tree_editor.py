# parse_editor_api.py
from flask import Flask, jsonify, request
from parse import FiniteParseTree, LanguageChunkingParser
from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
import json
import uuid

app = Flask(__name__)

LEARNING_ON = True
LOAD_LTM = ""
# LOAD_LTM = "unittests/gen_learn_test/final_ltm_data"

# --- Initialize parser and LTM ---
if LOAD_LTM != "":
    parser = LanguageChunkingParser.load_state(LOAD_LTM)
else:
    parser = LanguageChunkingParser(TEST_CORPUS2, context_length=2)

num_load = 0
document = []

for _ in range(num_load):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

for doc in document:
    parse_tree = parser.parse_input([doc], end_behavior="converge", debug=False)[0]
    parser.add_parse_tree(parse_tree, debug=False)

# --- Initialize first sentence and tree ---
sample_sentence = generate("S", TEST_GRAMMAR2)
# sample_sentence = "a man chases the woman"
curr_tree = FiniteParseTree(parser.get_long_term_memory(), parser.id_to_value, parser.value_to_id, context_length=2)
curr_tree.build_primitives(sample_sentence)
curr_tree._ensure_editor_state()

def reset_tree():
    """Refresh to a new sentence and rebuild current tree."""
    global curr_tree, sample_sentence
    sample_sentence = generate("S", TEST_GRAMMAR2)
    curr_tree = FiniteParseTree(parser.get_long_term_memory(), parser.id_to_value, parser.value_to_id, context_length=2)
    curr_tree.build_primitives(sample_sentence)
    curr_tree._ensure_editor_state()
    print(f"[INFO] New sentence selected: {sample_sentence}")

@app.route("/api/tree", methods=["GET"])
def api_get_tree():
    d3_json = curr_tree._draw_tree_to_json()
    pairs = curr_tree.get_parentless_pairs()
    return jsonify({
        "tree": d3_json,
        "pairs": pairs,
        "action_log": curr_tree.action_log,
        "sentence": sample_sentence
    })

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
        return jsonify({
            "ok": True,
            "tree": curr_tree._draw_tree_to_json(),
            "action_log": curr_tree.action_log,
            "apply_result": res
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/undo", methods=["POST"])
def api_undo():
    res = curr_tree.undo()
    return jsonify(res if isinstance(res, dict) else {
        "ok": res,
        "tree": curr_tree._draw_tree_to_json(),
        "action_log": curr_tree.action_log
    })

@app.route("/api/export", methods=["POST"])
def api_export():
    """Export current tree, add to LTM, then reset and refresh."""
    data = request.get_json() or {}
    filepath = data.get("filepath", "parse_tree_test.json")

    if filepath == "":
        filepath = "tree_" + str(uuid.uuid4())[:8]
    if not filepath.lower().endswith(".json"):
        filepath += ".json"

    # 1. Export to file
    export_path = f"gui/parse_tree_editor/{filepath}"
    res = curr_tree.export_json(export_path)

    # 2. Add to parser’s LTM
    if LEARNING_ON:
        parser.add_parse_tree(curr_tree, debug=False)

    # 3. Reset to a new random sentence
    reset_tree()

    # 4. Return message with refresh flag
    return jsonify({
        "ok": True,
        "message": f"Tree exported to {export_path}. LTM updated and new sentence loaded.",
        "refresh": True,
        "new_sentence": sample_sentence
    })

@app.route("/api/export_ltm", methods=["POST"])
def api_export_ltm():
    """
    Exports the entire parser long-term memory as JSON.
    Accepts optional 'filepath' in JSON body to save to disk.
    """
    data = request.get_json() or {}
    try:
        filepath = data.get("filepath")
    except TypeError:
        return jsonify({"ok": False})

    if not filepath or filepath == "":
        filepath = "ltm_" + str(uuid.uuid4())[:8]

    export_path = f"gui/parse_tree_editor/{filepath}"
    
    parser.save_state(export_path)
    parser.visualize_ltm(export_path)
    
    return jsonify({"ok": True, "filepath": export_path})

@app.route("/editor", methods=["GET"])
def editor_page():
    d3_json = json.dumps(curr_tree._draw_tree_to_json())
    html = curr_tree.editor_build_html(d3_json)
    return html

app.run(debug=True, port=5001)
