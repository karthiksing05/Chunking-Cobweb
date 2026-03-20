# parse_tree_editor_mh.py  –  Flask GUI for the multi-hierarchy framework
from flask import Flask, jsonify, request
from parse_mh import FiniteParseTree, WEBSTER, LongTermMemory
from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1, TEST_CORPUS2, TEST_GRAMMAR2
import json
import uuid

app = Flask(__name__)

LEARNING_ON = True
PREBUILD_TREES = False
CONTEXT_LENGTH = 3
THRESHOLD = -2
CATEGORIZATION_MODE = "dfs"  # "dfs", "bfs", or "bfs_pmi"
LOAD_LTM = ""
LOAD_LTM = "unittests/gen_learn_test_mh/final_ltm_data"

corpus = TEST_CORPUS1
grammar = TEST_GRAMMAR1

# --- Initialize WEBSTER (multi-hierarchy parser) ---
if LOAD_LTM != "":
    webster = WEBSTER.load_state(LOAD_LTM)
    # Override categorization_mode from config if loaded from state
    webster.categorization_mode = CATEGORIZATION_MODE
    webster.ltm.categorization_mode = CATEGORIZATION_MODE
    webster.content_bl_alpha = 1e-2
else:
    # Setting up the multi-hierarchy parser (WEBSTER)
    webster = WEBSTER(
        corpus,
        context_length=CONTEXT_LENGTH,
        threshold=THRESHOLD,
        content_alpha=1e-3,
        context_alpha=1e-3,
        content_bl_alpha=1,
        context_bl_alpha=1,
        bow=False,
        empty_weighting=True,
        weighting="binary",
        categorization_mode='dfs', # can be dfs, bfs, or bfs_pmi
        depth_max_content=1000,
        depth_max_context=1000,
        branch_max_content=1000,
        branch_max_context=1000,
    )

NUM_LOAD = 0
document = []

for _ in range(NUM_LOAD):
    sentence = generate("S", grammar)
    document.append(sentence)

for doc in document:
    webster.parse_sentence(doc, threshold=THRESHOLD, new_vocab=True, learning=True, debug=False)

# --- Initialize first sentence and tree ---
sample_sentence = generate("S", grammar)
curr_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
if PREBUILD_TREES:
    curr_tree.build(sample_sentence)
else:
    curr_tree.build_primitives(sample_sentence, threshold=THRESHOLD)


def reset_tree():
    """Refresh to a new sentence and rebuild current tree."""
    global curr_tree, sample_sentence
    sample_sentence = generate("S", grammar)
    curr_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    if PREBUILD_TREES:
        curr_tree.build(sample_sentence)
    else:
        curr_tree.build_primitives(sample_sentence, threshold=THRESHOLD)
    print(f"[INFO] New sentence selected: {sample_sentence}")


@app.route("/api/tree", methods=["GET"])
def api_get_tree():
    d3_json = curr_tree._draw_tree_to_json()
    pairs = curr_tree.get_parentless_pairs()
    from viz import CategorizePathVisualizer
    cpv = CategorizePathVisualizer()
    context_tree = cpv.tree_to_compact_json(webster.ltm.context_hierarchy.root)
    return jsonify({
        "tree": d3_json,
        "pairs": pairs,
        "action_log": curr_tree.action_log,
        "sentence": sample_sentence,
        "primitive_scores": curr_tree.get_primitive_score_data(),
        "context_tree": context_tree
    })


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    from viz import CategorizePathVisualizer
    data = request.get_json()
    left = data.get("left_word_index")
    right = data.get("right_word_index")
    debug = data.get("debug", True)
    result = curr_tree.evaluate_pair(left, right, debug=debug)
    cpv = CategorizePathVisualizer()
    content_tree = cpv.tree_to_compact_json(webster.ltm.content_hierarchy.root)
    context_tree = cpv.tree_to_compact_json(webster.ltm.context_hierarchy.root)
    return jsonify({"ok": True, "result": result,
                    "content_tree": content_tree, "context_tree": context_tree})


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
    export_path = f"gui/parse_tree_editor_mh/{filepath}"
    res = curr_tree.export_json(export_path)

    # 2. Add to LTM (both content + context hierarchies)
    if LEARNING_ON:
        webster.ltm.add_parse_tree(curr_tree, shuffle=True, debug=False)

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
    Exports the entire multi-hierarchy LTM (content + context).
    Accepts optional 'filepath' in JSON body to save to disk.
    """
    data = request.get_json() or {}
    try:
        filepath = data.get("filepath")
    except TypeError:
        return jsonify({"ok": False})

    if not filepath or filepath == "":
        filepath = "ltm_" + str(uuid.uuid4())[:8]

    export_path = f"gui/parse_tree_editor_mh/{filepath}"

    webster.save_state(export_path)
    webster.visualize_ltm(export_path, max_depth=4)

    return jsonify({"ok": True, "filepath": export_path})


@app.route("/editor", methods=["GET"])
def editor_page():
    d3_json = json.dumps(curr_tree._draw_tree_to_json())
    html = curr_tree.editor_build_html(d3_json)
    return html


app.run(debug=True, port=5001)
