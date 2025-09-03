import uuid
import os
import json
import asyncio
from playwright.async_api import async_playwright
import re
from util.pycobweb import CobwebTree, CobwebNode
from viz import HTMLCobwebDrawer
from typing import List
from sortedcontainers import SortedList


"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class PrimitiveParseNode:

    def __init__(self, content, context_before, context_after, word_index):

        self.global_root = False
        self.word_index = word_index

        self.parent = None
        self.children = SortedList()

        self.title = uuid.uuid4().hex[:10] # random id

        self.content = content
        self.context_before = context_before
        self.context_after = context_after

        self.concept_label = None

    def set_parent(self, node):
        """
        Helper method to set the parent of the current node!
        We're going to adjust both parents and children with this method, to
        allow for efficient designation.

        This method also deletes the existing parent-child connection, if it
        already exists (to efficiently manage the root node changes).

        TODO one change to make here may be developing a more finetuned version
        of this because we only need to remove and reassign the global root node
        as a parent.
        """

        try:
            self.parent.children.remove((self.word_index, self))
        except AttributeError: # trying to assign parentship for the first time
            # print("Assigning current node's parent as the global root node for brevity")
            pass
        except ValueError: # this should never happen
            print("Parent does not exist or parent does not include the current node as its child")

        self.parent = node
        node.children.add((self.word_index, self))

    def get_as_instance(self):
        """
        Helper method to get the current parse node as an instance description!
        """

        return {
            "content": self.content,
            "context-before": self.context_before,
            "context-after": self.context_after
        }

        # return {
        #     0: self.content_left,
        #     1: self.content_right,
        #     2: self.context_before,
        #     3: self.context_after
        # }

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class CompositeParseNode:

    def __init__(self):

        self.global_root = False
        self.word_index = None

        self.parent = None
        self.children = SortedList()

        self.title = uuid.uuid4().hex[:10] # random id

        self.content_left = None
        self.content_right = None
        self.context_before = None
        self.context_after = None

        self.concept_label = None

    @staticmethod
    def create_global_root():
        """
        A static method that creates an empty root node to act as the navigation
        for all parents. This will be extremely useful for keeping track of
        partial parses with our cutoff.
        """

        node = CompositeParseNode()

        node.global_root = True

        return node

    @staticmethod
    def create_node(instance_dict, closest_concept_id, word_index):
        """
        One of the method-based constructors to create the correct version
        of our parse nodes at the leaf level - this will create a parse node
        from a directly-parsed instance from the input.
        """

        node = CompositeParseNode()

        node.content_left = instance_dict[0]
        node.content_right = instance_dict[1]

        node.context_before = instance_dict[2]
        node.context_after = instance_dict[3]

        node.concept_label = closest_concept_id

        node.word_index = word_index

        return node

    @staticmethod
    def create_merge_instance(node_left, node_right):
        """
        One of the method-based constructors to create the correct version
        of our parse nodes at the not leaf level - it should properly reference
        context and construct a parse node based on two given parse nodes.

        NOTE: this is not going to actively create a node yet, it's a helper
        method to merge instances. We'll take the best of these instances (from
        the categorization) and then add it to our parse tree.

        CONTEXT CLARIFICATION:
        I would assume that the label for a given node is in terms of the
        concept or instance that it references within the code. Most initial
        instances will be pretty useless / have useless concept labels, but
        eventually we'll be able to track stronger correlations!

        CONTENT CLARIFICATION:
        For content, we have one of two clarifications (which I've asked Pat
        about).
        - https://docs.google.com/presentation/d/1k1PNL73OuZC2lCdqy-q-OfNPMiYJlft04t_kOJTNUZ0/edit?slide=id.g3719c2a0e40_0_26#slide=id.g3719c2a0e40_0_26
        The above URL summarizes it best - currently working with Option 2,
        which we're going to implement, but either should be solid. For brevity,
        Option 2 involves using only primitive instances to represent content
        and context, which is easily done for our tree.
        """

        new_inst_dict = {}

        if type(node_left) == PrimitiveParseNode:
            new_inst_dict[0] = {node_left.content: 1}
        else:
            new_inst_dict[0] = {node_left.concept_label: 1}

        if type(node_right) == PrimitiveParseNode:
            new_inst_dict[1] = {node_right.content: 1}
        else:
            new_inst_dict[1] = {node_right.concept_label: 1}

        # we need to import the previous node's content and context
        # I have reason to think that the below code will just work LOL
        # BUT subject to change!
        new_inst_dict[2] = node_left.context_before
        new_inst_dict[3] = node_right.context_after

        return new_inst_dict

    def set_parent(self, node):
        """
        Helper method to set the parent of the current node!
        We're going to adjust both parents and children with this method, to
        allow for efficient designation.

        This method also deletes the existing parent-child connection, if it
        already exists (to efficiently manage the root node changes).

        TODO one change to make here may be developing a more finetuned version
        of this because we only need to remove and reassign the global root node
        as a parent.
        """

        try:
            self.parent.children.remove((self.word_index, self))
        except AttributeError: # trying to assign parentship for the first time
            # print("Assigning current node's parent as the global root node for brevity")
            pass
        except ValueError: # this should never happen
            print("Parent does not exist or parent does not include the current node as its child")

        self.parent = node
        node.children.add((self.word_index, self))

    def get_as_instance(self):
        """
        Helper method to get the current parse node as an instance description!
        """

        return {
            0: self.content_left,
            1: self.content_right,
            2: self.context_before,
            3: self.context_after
        }

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class ParseTree:

    def __init__(self, ltm_hierarchy, id_to_value, value_to_id, context_length=3):
        self.ltm_hierarchy = ltm_hierarchy

        self.id_to_value = id_to_value
        self.value_to_id = value_to_id

        self.context_length = context_length

        self.global_root_node = CompositeParseNode.create_global_root()

        self.sentence = None

        self.nodes = []

    def build(self, sentence, end_behavior="converge", debug=False): # TODO debug flag needs to be implemented
        """
        Primary method of construction that returns all available nonterminals
        as instances ready to be passed into the long-term hierarchy. The
        following process takes place:
        *   First, all chunk candidates are proposed based on the current set of
            non-terminals without parents.
        *   Each candidate is categorized within the Cobweb hierarchy, and the
            candidate that finds the best fit is added to the parse tree.
            *   I believe Chris summarized this well as the chunk that finds the
                concept that has the highest probability of generating that chunk is
                added to the parse tree.

        IMPORTANT PARAM: "end_behavior" --> can be "converge" to represent the
        tree converging on one root or a float to represent the tree continually
        updating until no candidates proposed have a valuable enough addition.

        IMPORTANT PARAM: "content_ids" --> can be "longterm" to represent that composite nodes,
        after content, can inherit from localized IDs

        Process for creating a non-terminal parse node:
        *   initialize the parse node based on an instance dictionary
        *   assign it its respective concept label
        *   because it's a new parse node, connect it to the global root node and
            its two children nodes

        Returns:
        *   the list of new concept labels that are required
        """

        self.sentence = sentence

        elements = re.findall(r"[\w']+|[.,!?;]", sentence)
        elements = [self.value_to_id[element] for element in elements]

        # Creating first layer of primitive nodes
        for i in range(len(elements)):

            content = elements[i]

            context_before_lst = elements[max(0, i - self.context_length):(i)][::-1]
            context_after_lst = elements[(i + 1):min(len(elements), i + self.context_length + 1)]

            # have to compute the dictionaries through a for loop so that we can
            # add multiple weights for multiple instances of the word

            context_before_dict = dict()
            context_after_dict = dict()

            for j in range(len(context_before_lst)):
                context_before_dict.setdefault(context_before_lst[j], 0)
                context_before_dict[context_before_lst[j]] += 1 / (j + 1) # this works because we reversed it above

            for j in range(len(context_after_lst)):
                context_after_dict.setdefault(context_after_lst[j], 0)
                context_after_dict[context_after_lst[j]] += 1 / (j + 1)

            node = PrimitiveParseNode(content, context_before_dict, context_after_dict, i)

            node.set_parent(self.global_root_node)

            self.nodes.append(node)

        # Creating other layers of composite nodes
        while True:

            best_candidate = None # stores concepts in the form of (score, label, node_left, node_right)

            parentless = [x[1] for x in self.global_root_node.children]
            # print([(x[0], x[1].get_as_instance()) for x in self.global_root_node.children])

            # TODO one change we can make here is constantly keeping a candidate list and then not
            # worry about recomputing everything -> leaving this here for now but subject to change!

            for i in range(len(parentless) - 1):
                node_left = parentless[i]
                node_right = parentless[i + 1]

                merge_inst = CompositeParseNode.create_merge_instance(node_left, node_right)

                candidate_concept = self.ltm_hierarchy.categorize(merge_inst).get_basic_level() # TODO can also change this to get "best_level"
                candidate_concept_id = self.value_to_id[f"CONCEPT-{candidate_concept.concept_hash()}"]

                candidate_score = candidate_concept.log_prob_instance(merge_inst)

                if debug:
                    print("MERGE INSTANCE EVALUATED: ", merge_inst)
                    # the below thing always returns -inf for some reason!??
                    # print("Candidate Concept Log Prob Children Given Instance", 
                    #       candidate_concept.log_prob_children_given_instance(merge_inst))
                    print("Candidate Concept Log Prob Instance", 
                          candidate_concept.log_prob_instance(merge_inst))
                    print("Candidate Concept Log Prob Instance Missing", 
                          candidate_concept.log_prob_instance_missing(merge_inst))
                    print("Candidate Concept Log Prob Class Given Instance", 
                          candidate_concept.log_prob_class_given_instance(merge_inst))
                    print()

                # TODO IMPLEMENT A LOG PROBABILITY SCORE INSTEAD OF A CU SCORE
                if not best_candidate or candidate_score > best_candidate[0]:
                    if debug:
                        print(f"New best candidate found with log-probability {candidate_score} for concept hash {candidate_concept.concept_hash()}")
                        print()
                    best_candidate = (candidate_score, candidate_concept_id, node_left, node_right)

            if isinstance(end_behavior, (int, float)) and best_candidate[0] < end_behavior:
                if debug:
                    print("BEST MERGE CANDIDATE was not good enough to be passed, only score of", best_candidate[0])
                    print("---------" * 7)
                break

            best_merge_inst = CompositeParseNode.create_merge_instance(best_candidate[2], best_candidate[3])

            if debug:
                print("BEST MERGE CANDIDATE FOUND: ", best_merge_inst)
                print("with score, ", best_candidate[0])
                print("---------" * 7)

            candidate_concept = self.ltm_hierarchy.categorize(best_merge_inst).get_basic_level()
            candidate_concept_id = self.value_to_id[f"CONCEPT-{candidate_concept.concept_hash()}"]

            add_parse_node = CompositeParseNode.create_node(best_merge_inst, candidate_concept_id, 0.5 * (best_candidate[2].word_index + best_candidate[3].word_index))

            # changing parentship!
            self.nodes.append(add_parse_node)
            add_parse_node.set_parent(self.global_root_node)
            best_candidate[2].set_parent(add_parse_node)
            best_candidate[3].set_parent(add_parse_node)

            if len(self.global_root_node.children) == 1:
                break

        return True

    def get_chunk_instances(self):
        """
        Primary method that returns all available nonterminals as instances
        ready to be passed into the long-term hierarchy
        """

        instances = []

        def dfs_insts(node):
            if type(node) == PrimitiveParseNode:
                return

            instances.append(node.get_as_instance())

            for _, child in node.children:
                dfs_insts(child)

        for _, child in self.global_root_node.children:
            dfs_insts(child)

        return instances

    def visualize(self, out_base="parse_tree", render_png=True):
        """
        Render the parse tree into an HTML file and optionally a PNG screenshot.
        The PNG height automatically adjusts to fit the tree, including all nodes.
        """

        # Convert tree to JSON and build HTML
        d3_json = json.dumps(self._tree_to_json())
        html = self._build_html(d3_json)

        html_path = f"{out_base}.html"
        png_path = f"{out_base}.png"

        # Ensure output directories exist
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

        # Write HTML file
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        if render_png:
            asyncio.run(self._html_to_png(html_path, png_path))
            return html_path, png_path
        else:
            return html_path

    async def _html_to_png(self, html_path, png_path):
        """
        Convert the HTML tree to a PNG screenshot using Playwright.
        Automatically adjusts the PNG size to fit the full tree height.
        """

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("file://" + os.path.abspath(html_path))

            # Wait until the tree SVG exists
            await page.wait_for_selector("#tree svg")

            # Wait for layout of dynamic node content to stabilize
            await page.evaluate("""
                () => new Promise(resolve => {
                    requestAnimationFrame(() => requestAnimationFrame(resolve));
                })
            """)

            # Measure the wrapper container including all overflow from foreignObjects
            bounding_box = await page.evaluate("""
                () => {
                    const container = document.querySelector('#tree-container');
                    return {
                        width: Math.ceil(container.scrollWidth) + 20,
                        height: Math.ceil(container.scrollHeight) + 20
                    };
                }
            """)

            # Set viewport to actual content size
            await page.set_viewport_size({
                "width": bounding_box["width"],
                "height": bounding_box["height"]
            })

            # Take a screenshot of the SVG
            svg_elem = await page.query_selector("#tree svg")
            await svg_elem.screenshot(path=png_path, scale="css")

            await browser.close()

    def _safe_lookup(self, idx):
        if (idx is not None and 0 <= idx < len(self.id_to_value)):
            return self.id_to_value[idx]
        else:
            # print("index", idx)
            return "None"

    def _node_to_dict(self, node, children_getter):
        def ctx_list(ctx):
            if not ctx:
                return []
            items = sorted(ctx.items(), key=lambda kv: (-kv[1], kv[0]))
            return [{"key": self._safe_lookup(k), "val": float(v)} for k, v in items]

        if isinstance(node, PrimitiveParseNode):
            return {
                "title": node.title,
                "left": self._safe_lookup(node.content),
                "right": None,
                "before": ctx_list(node.context_before),
                "after": ctx_list(node.context_after),
                "children": [self._node_to_dict(ch[1], children_getter) for ch in children_getter(node)]
            }

        elif isinstance(node, CompositeParseNode):
            left_id  = None if not node.content_left  else next(iter(node.content_left.keys()))
            right_id = None if not node.content_right else next(iter(node.content_right.keys()))
            left  = self._safe_lookup(left_id)
            right = self._safe_lookup(right_id)

            return {
                "title": node.title,
                "left": left,
                "right": right,
                "before": ctx_list(node.context_before),
                "after":  ctx_list(node.context_after),
                "children": [self._node_to_dict(ch[1], children_getter) for ch in children_getter(node)]
            }

        else:
            raise TypeError(f"Unknown node type {type(node)}")


    def _tree_to_json(self):
        def children_getter(n):
            for wi, ch in getattr(n, "children", []):
                yield (wi, ch)
        return self._node_to_dict(self.global_root_node, children_getter)

    def _build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>Parse Tree</title>
    <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    #tree-container {{ display: inline-block; }}
    .link {{ fill: none; stroke: #9aa1a9; stroke-width: 1.5px; }}
    .node-box {{ stroke: #444; fill: #fff; rx: 8; ry: 8; filter: drop-shadow(1px 2px 2px rgba(0,0,0,0.15)); }}
    .node-fo table {{ border-collapse: collapse; font-size: 12px; margin: 4px 0; }}
    .node-fo th, .node-fo td {{ border: 1px solid #888; padding: 2px 6px; }}
    .node-fo th {{ background: #f3f5f7; font-weight: 600; }}
    .section-title {{ font-weight: bold; margin-top: 4px; }}
    .section {{ margin-top: 10px; margin-bottom: 10px; }}
    .subtable b {{ display: inline-block; margin: 6px 0 2px; }}
    .subtable table {{ border-collapse: collapse; }}
    .subtable td {{ border: 1px solid #bbb; padding: 1px 4px; }}
    </style>
    </head>
    <body>
    <div id="tree-container">
    <div id="tree"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
    const data = {d3_data_json};
    const nodeW  = {node_w};
    const nodeH  = {node_h};
    const hGap   = {h_gap};
    const vGap   = {v_gap};

    const root = d3.hierarchy(data);
    const layout = d3.tree().nodeSize([nodeW + hGap, nodeH + vGap]);
    layout(root);

    // compute bounds
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    root.each(d => {{
        if (d.x < x0) x0 = d.x;
        if (d.x > x1) x1 = d.x;
        if (d.y < y0) y0 = d.y;
        if (d.y > y1) y1 = d.y;
    }});
    const width  = x1 - x0 + nodeW + 320;
    const height = y1 - y0 + nodeH + 320; // THIS IS THE HEIGHT MODIFIER

    const svg = d3.select("#tree").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [x0 - nodeW/2, y0 - nodeH/2, width, height].join(" "));

    const g = svg.append("g");

    // links
    g.selectAll("path.link")
    .data(root.links())
    .join("path")
    .attr("class", "link")
    .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

    // nodes
    const node = g.selectAll("g.node")
    .data(root.descendants())
    .join("g")
    .attr("transform", d => `translate(${{d.x}},${{d.y}})`);

    // node rect
    node.append("rect")
    .attr("class", "node-box")
    .attr("x", -nodeW/2)
    .attr("y", 0)
    .attr("width", nodeW)
    .attr("height", nodeH);

    // node HTML via foreignObject
    node.append("foreignObject")
    .attr("class", "node-fo")
    .attr("x", -nodeW/2 + 6)
    .attr("y", 6)
    .attr("width", nodeW - 12)
    .attr("height", 1000)
    .html(d => nodeHTML(d.data));

    // shrink foreignObjects to actual content height
    node.selectAll("foreignObject").each(function() {{
    const fo = d3.select(this);
    const div = fo.select("div").node();
    const h = div.getBoundingClientRect().height + 6;
    fo.attr("height", h);
    d3.select(this.parentNode).select("rect").attr("height", h + 12);
    }});

    function nodeHTML(d) {{
        const ctxTable = (ctx, title) => {{
            if (!ctx || ctx.length === 0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
            const rows = ctx.map(kv => `<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
            return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
        }};

        let contentRows = "";
        if (d.right && d.right !== "None") {{
            // Composite node with left and right
            contentRows = `
                <tr><td>Content-Left</td><td>${{d.left}}</td></tr>
                <tr><td>Content-Right</td><td>${{d.right}}</td></tr>`;
        }} else {{
            // Primitive node with single content
            contentRows = `<tr><td>Content</td><td>${{d.left}}</td></tr>`;
        }}

        return `
        <div class="node-fo">
            <table><tr><th colspan="2">${{d.title}}</th></tr></table>
            <table>
            ${{contentRows}}
            </table>
            ${{ctxTable(d.before, "Context-Before")}}
            ${{ctxTable(d.after,  "Context-After")}}
        </div>`;
    }}

    </script>
    </body>
    </html>
    """


    def to_json(self, filepath=None):
        """
        Serialize the ParseTree into JSON. Optionally save to `filepath`.
        """

        def serialize_node(node, index_map):
            if isinstance(node, PrimitiveParseNode):
                return {
                    "node_type": "primitive",
                    "title": node.title,
                    "word_index":node.word_index,
                    "content": node.content,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "concept_label": node.concept_label,
                    "global_root": node.global_root,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            elif isinstance(node, CompositeParseNode):
                return {
                    "node_type": "composite",
                    "title": node.title,
                    "word_index":node.word_index,
                    "content_left": node.content_left,
                    "content_right": node.content_right,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "concept_label": node.concept_label,
                    "global_root": node.global_root,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            else:
                raise TypeError(f"Unknown node type {type(node)}")

        index_map = {node: i for i, node in enumerate([self.global_root_node] + self.nodes)}
        data = {
            "sentence": self.sentence,
            "context_length": self.context_length,
            "id_to_value": self.id_to_value,
            "value_to_id": self.value_to_id,
            "nodes": [serialize_node(node, index_map) for node in [self.global_root_node] + self.nodes],
        }

        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return filepath
        else:
            return json.dumps(data, indent=2)

    @staticmethod
    def from_json(data, ltm_hierarchy, filepath=False):
        """
        Deserialize a ParseTree from JSON. Requires the same ltm_hierarchy instance.
        """
        if filepath:
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(data, str):
            data = json.loads(data)

        tree = ParseTree(
            ltm_hierarchy,
            id_to_value=data["id_to_value"],
            value_to_id=data["value_to_id"],
            context_length=data["context_length"],
        )
        tree.sentence = data["sentence"]

        def restore_dict_keys(d):
            if d is None:
                return None
            return {int(k): v for k, v in d.items()}

        node_objs = []
        for n in data["nodes"]:
            if n["node_type"] == "primitive":
                node = PrimitiveParseNode(
                    content=n["content"],
                    context_before=restore_dict_keys(n["context_before"]),
                    context_after=restore_dict_keys(n["context_after"]),
                    word_index=n["word_index"],
                )
            elif n["node_type"] == "composite":
                node = CompositeParseNode()
                node.content_left = restore_dict_keys(n["content_left"])
                node.content_right = restore_dict_keys(n["content_right"])
                node.context_before = restore_dict_keys(n["context_before"])
                node.context_after = restore_dict_keys(n["context_after"])
                node.word_index = n["word_index"]

            else:
                raise ValueError(f"Unknown node_type {n['node_type']}")

            node.title = n["title"]
            node.concept_label = n["concept_label"]
            node.global_root = n["global_root"]
            node_objs.append(node)

        # restore parent/child relations
        for idx, n in enumerate(data["nodes"]):
            node = node_objs[idx]
            parent_idx = n["parent"]
            if parent_idx is not None:
                node.parent = node_objs[parent_idx]
            node.children = [(node_objs[ch].word_index, node_objs[ch]) for ch in n["children"]]

        tree.global_root_node = node_objs[0]
        tree.nodes = node_objs[1:]

        return tree

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class LanguageChunkingParser:

    """
    This is only for language right now - lots of changes will need to be made
    to generalize, such as properly representing context.

    Standardizations:
    *   id-0 => content-left
    *   id-1 => content-right
    *   id-2 => context-before
    *   id-3 => context-after
    """

    def __init__(self, value_corpus, context_length=3, merge_split=True):

        self.ltm_hierarchy = CobwebTree()

        self.id_to_value = [x for x in value_corpus]
        self.value_to_id = dict([(w, i) for i, w in enumerate(value_corpus)])
        self.id_count = len(value_corpus)

        self.cobweb_drawer = HTMLCobwebDrawer(
            ["Content-Left", "Content-Right", "Context-Before", "Context-After"],
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id
        )

        # adding root node to dictionary! edge case not properly counted
        hash = self.ltm_hierarchy.root.concept_hash()
        self.value_to_id[f"CONCEPT-{hash}"] = self.id_count
        self.id_to_value.append(f"CONCEPT-{hash}")
        self.id_count += 1

        self.context_length = context_length
        self.merge_split = merge_split

    def get_long_term_memory(self):
        return self.ltm_hierarchy

    def parse_input(self, sentences, end_behavior="converge", debug=False) -> List[ParseTree]:
        """
        Primary method for parsing input (a list of sentences) and updating the
        long-term-memory hierarchy using a parse tree.

        Returns the Parse Tree!
        """

        parse_trees = []

        for sentence in sentences:
            if debug:
                print(f"BUILDING PARSE TREE FOR SENTENCE {sentence}")
            parse_tree = ParseTree(self.ltm_hierarchy, self.id_to_value, self.value_to_id)
            parse_tree.build(sentence, end_behavior, debug)

            parse_trees.append(parse_tree)

            if debug:
                print("-" * 100)
                print()

        return parse_trees

    def add_parse_tree(self, parse_tree, debug=False):
        """
        Method to add the parse tree to the long-term hierarchy.

        Important note here is that because the parse tree doesn't contain the
        primitive instances in code (the leaves of the code tree are composite
        by nature), we can literally just add all nodes of the parse tree as
        defined by the code.

        Pipeline:
        *   Save tree structure before adding all new nodes (for each node,
            what is its parent and what are all its children)
        *   Add all instances of the parse tree as children via Cobweb's ifit
        *   Now, we iterate over the new tree and the old tree and keep track of
            all fundamentally applied actions.
        *   We initialize an empty list, "rewrite_rules", for updating all
            instance dictionaries after properly logging all actions.
        *   We process each Cobweb action in an effort to keep an updated as
            follows:
            *   Any "Add" operations don't change the tree's node structure and
                are not necessary to process.
            *   Any "Create" operation creates a new node (typically the last
                action) - these nodes must be added via the pattern
                "CONCEPT-{node.concept_hash()}" to the self.value_to_id and
                self.id_to_value dictionaries (also update self.id_count).
            *   Any "Split" operation deletes a node and promotes all children
                to become parents - I'm PRETTY SURE there's no vocabulary
                changes, but we will need to save a rewrite rule binding the old
                node vocabulary id to it's parent's id, (deleted_id, parent_id).
            *   Any "Merge" operation groups two children and creates a common
                parent node for these two nodes. In this case, we should add the
                new node to the vocabulary similar to the "Create" action.
        *   Finally, we iterate through the tree and apply all rewrite rules
            (which I'm pretty sure are only SPLIT rules, so this should be
            fairly easy).

        Additional Notes:
        *   We're currently just comparing the old tree to the new tree to
            safely make all changes, but a future implementation will rewrite
            the Cobweb class to natively return these actions for our
            convenience and faster time-complexity.
        *   In the long term, we should keep track of "stale" concept ids in our
            vocabulary to replace them over time so our vocabulary doesn't
            balloon, but again, a later fix.
        *   Finally, rewrite rules may not currently exist for created nodes but
            there is a layer of specificity that we can and should replace.
            Still not sure about the best way to go about that, but leaving it
            here for future notice.
        """

        if debug:
            print(f"Adding parse tree for sentence, \"{parse_tree.sentence}\"")

        # adding all new instances
        insts = parse_tree.get_chunk_instances()

        if not self.merge_split:
            all_actions = []
            for inst in insts:
                if debug:
                    print("Adding instance to CobwebTree:", inst)
                # self.cobweb_drawer.draw_tree(self.ltm_hierarchy.root, "tests/gen_learn_test/test_trees/test_parse_tree")
                _, _, actions = self.ltm_hierarchy.ifit(inst, mode=4)
                all_actions += actions

            if debug:
                print("PRINTING ALL ACTIONS TAKEN OVER THIS PASS:")
                for i, act in enumerate(actions, 1):
                    a = act["action"]
                    node = act.get("node")
                    parent = act.get("parent")
                    extras = act.get("extra_nodes_created", [])
                    absorbed = act.get("absorbed", False)

                    # Format nicely by action type
                    if a == "NEW":
                        if absorbed:
                            print(f"{i:02d}. [NEW/ABSORB] Node {node} absorbed instance "
                                f"(parent={parent or 'ROOT'})")
                        elif extras:
                            print(f"{i:02d}. [NEW] Node {node} (parent={parent}) "
                                f"(extra created: {', '.join(extras)})")
                        else:
                            print(f"{i:02d}. [NEW] Node {node} (parent={parent})")

                    elif a == "BEST":
                        print(f"{i:02d}. [BEST] Descend into node {node} (parent={parent})")

                    elif a == "MERGE":
                        kids = ", ".join(act.get("children", []))
                        print(f"{i:02d}. [MERGE] New node {act['new_node']} under {act['parent']} "
                            f"merged children [{kids}]")

                    elif a == "SPLIT":
                        kids = ", ".join(act.get("promoted_children", []))
                        print(f"{i:02d}. [SPLIT] Deleted {act['deleted']} under {act['parent']}, "
                            f"promoted [{kids}]")

                    elif a == "FALLBACK":
                        print(f"{i:02d}. [FALLBACK] Went to {node} (parent={parent})")

                    else:
                        print(f"{i:02d}. [UNKNOWN] {act}")

                print("---")

            # NEED TO ADD NEW NODES NO MATTER WHAT
            for action in all_actions:
                if action["action"] == "NEW":
                    # ADD CREATED NODES TO VOCABULARY
                    new_vocab = f"CONCEPT-{action["node"]}"
                    self.value_to_id[new_vocab] = self.id_count
                    self.id_to_value.append(new_vocab)
                    self.id_count += 1

                    if "extra_nodes_created" in action:
                        for con_hash in action["extra_nodes_created"]:
                            new_vocab = f"CONCEPT-{con_hash}"
                            if new_vocab not in self.value_to_id:
                                self.value_to_id[new_vocab] = self.id_count
                                self.id_to_value.append(new_vocab)
                                self.id_count += 1
                elif action["action"] == "MERGE":
                    # ADD CREATED NODES TO VOCABULARY
                    new_vocab = f"CONCEPT-{action["new_node"]}"
                    self.value_to_id[new_vocab] = self.id_count
                    self.id_to_value.append(new_vocab)
                    self.id_count += 1

        else:
            """
            With this new method, we have a couple administrative actions we need 
            to complete, but this should greatly reduce the size of the code by a
            large amount (and the lasting complexity as well).

            After adding to the tree, we'll have a stack-trace of actions. We need
            to conduct the following two administrative actions:
            *   All new nodes need to be added to the vocabulary
            *   All deleted nodes need to be transferred to a rewrite-rules list 
                and a new method needs to handle all rewrites to the cobweb tree
                recursively.

            We will implement both of these in here so as not to touch the
            administrative Cobweb.

            For created nodes:
            *   If the node is created:
                *   Add it to the vocabulary!
            For deleted nodes:
            *   If the node is deleted:
                *   Iterate through all nodes and replace the node's id with its
                    parent's id.

            TODO: Note that at some point, we'll need to program a fix that overrides
            av_count of the child as well as the recursive path taken to the root.
            We can probably program some cool BFS from the root that adds the probability
            of old_key to new_key and then removes old_key, and then traverses down all
            children, stopping the traversal if old_key doesn't exist.
            """
            all_actions = []

            for inst in insts:
                if debug:
                    print("Adding instance to CobwebTree:", inst)
                # self.cobweb_drawer.draw_tree(self.ltm_hierarchy.root, "tests/gen_learn_test/test_trees/test_parse_tree")
                _, _, actions = self.ltm_hierarchy.ifit(inst, mode=0)
                all_actions += actions

                if debug:
                    print("PRINTING ALL ACTIONS TAKEN OVER THIS PASS:")
                    for i, act in enumerate(actions, 1):
                        a = act["action"]
                        node = act.get("node")
                        parent = act.get("parent")
                        extras = act.get("extra_nodes_created", [])
                        absorbed = act.get("absorbed", False)

                        # Format nicely by action type
                        if a == "NEW":
                            if absorbed:
                                print(f"{i:02d}. [NEW/ABSORB] Node {node} absorbed instance "
                                    f"(parent={parent or 'ROOT'})")
                            elif extras:
                                print(f"{i:02d}. [NEW] Node {node} (parent={parent}) "
                                    f"(extra created: {', '.join(extras)})")
                            else:
                                print(f"{i:02d}. [NEW] Node {node} (parent={parent})")

                        elif a == "BEST":
                            print(f"{i:02d}. [BEST] Descend into node {node} (parent={parent})")

                        elif a == "MERGE":
                            kids = ", ".join(act.get("children", []))
                            print(f"{i:02d}. [MERGE] New node {act['new_node']} under {act['parent']} "
                                f"merged children [{kids}]")

                        elif a == "SPLIT":
                            kids = ", ".join(act.get("promoted_children", []))
                            print(f"{i:02d}. [SPLIT] Deleted {act['deleted']} under {act['parent']}, "
                                f"promoted [{kids}]")

                        elif a == "FALLBACK":
                            print(f"{i:02d}. [FALLBACK] Went to {node} (parent={parent})")

                        else:
                            print(f"{i:02d}. [UNKNOWN] {act}")

                    print("---")


            # SEPARATE ACTIONS INTO CREATED AND DELETED
            rewrite_rules = [] # (old, new)

            for action in all_actions:
                if action["action"] == "NEW":
                    # ADD CREATED NODES TO VOCABULARY
                    new_vocab = f"CONCEPT-{action["node"]}"
                    self.value_to_id[new_vocab] = self.id_count
                    self.id_to_value.append(new_vocab)
                    self.id_count += 1

                    if "extra_nodes_created" in action:
                        for con_hash in action["extra_nodes_created"]:
                            new_vocab = f"CONCEPT-{con_hash}"
                            if new_vocab not in self.value_to_id:
                                self.value_to_id[new_vocab] = self.id_count
                                self.id_to_value.append(new_vocab)
                                self.id_count += 1
                elif action["action"] == "MERGE":
                    # ADD CREATED NODES TO VOCABULARY
                    new_vocab = f"CONCEPT-{action["new_node"]}"
                    self.value_to_id[new_vocab] = self.id_count
                    self.id_to_value.append(new_vocab)
                    self.id_count += 1
                elif action["action"] == "SPLIT":
                    # ADD TO REWRITE RULES
                    rewrite_rules.append((action["deleted"], action["parent"]))

            # BFS/DFS THROUGHOUT TREE AND EDIT AV_COUNTS WHERE APPLICABLE
            def av_replacement(inst):
                replaced = False

                for k in inst.keys():
                    for concept_hash in inst[k].keys():
                        for rewrite in rewrite_rules:
                            if f"CONCEPT-{concept_hash}" == rewrite[0]:
                                inst[k].setdefault(f"CONCEPT-{rewrite[1]}", 0)
                                inst[k][rewrite[1]] += inst[k][rewrite[0]]
                                del inst[k][rewrite[0]]
                                replaced = True

                return inst, replaced


            to_visit = [self.ltm_hierarchy.root]

            while len(to_visit) > 0:
                curr = to_visit.pop(0)
                new_av_count, replaced = av_replacement(curr.av_count)
                curr.av_count = new_av_count

                if replaced:
                    for child in curr.children:
                        to_visit.append(child)

        if debug:
            print("-" * 60)

        return True

    def visualize_ltm(self, out_base="cobweb_tree"):
        """
        We had a rudimentary CobwebDrawer before but I'd very much enjoy if we
        could expand on this and create an HTML-drawing Cobweb method before we
        continue tests - it would be both easier to explain and certainly easy
        to verify.
        """
        self.cobweb_drawer.draw_tree(self.ltm_hierarchy.root, out_base)