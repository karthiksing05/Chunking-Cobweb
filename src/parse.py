import secrets
import json, os, asyncio, nest_asyncio
from IPython.display import Image, display
from pyppeteer import launch
import heapq
import json
import re

nest_asyncio.apply()  # allow nested event loops in Jupyter

class ParseNode:

    def __init__(self):

        self.global_root = False

        self.parent = None
        self.children = []

        self.title = str(secrets.token_hex(4)) # random id

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

        node = ParseNode()

        node.global_root = True

        return node

    @staticmethod
    def create_node(instance_dict, closest_concept_id):
        """
        One of the method-based constructors to create the correct version
        of our parse nodes at the leaf level - this will create a parse node
        from a directly-parsed instance from the input.
        """

        node = ParseNode()

        node.content_left = instance_dict[0]
        node.content_right = instance_dict[1]

        node.context_before = instance_dict[2]
        node.context_after = instance_dict[3]

        node.concept_label = closest_concept_id

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

        new_inst_dict[0] = {node_left.concept_label: 1}
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
            self.parent.children.remove(self)
        except AttributeError: # trying to assign parentship for the first time
            # print("Assigning current node's parent as the global root node for brevity")
            pass
        except ValueError: # this should never happen
            print("Parent does not exist or parent does not include the current node as its child")

        self.parent = node
        node.children.append(self)

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
"""

class ParseTree:

    def __init__(self, ltm_hierarchy, id_to_value, value_to_id, context_length=3):
        self.ltm_hierarchy = ltm_hierarchy

        self.id_to_value = id_to_value
        self.value_to_id = value_to_id

        self.context_length = context_length

        self.global_root_node = ParseNode.create_global_root()

        self.sentence = None

        self.nodes = []

    def build(self, sentence, end_behavior="converge", debug=False):
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

        Process for creating a non-terminal parse node:
        *   initialize the parse node based on an instance dictionary
        *   assign it its respective concept label
        *   because it's a new parse node, connect it to the global root node and
            its two children nodes

        Returns:
        *   the list of new concept labels that are required
        """

        self.sentence = sentence

        instances = []

        elements = [self.value_to_id[element] for element in re.findall(r"[\w']+|[.,!?;]", sentence)]

        for i in range(len(elements) - 1):

            content_left = elements[i]
            content_right = elements[i + 1]

            context_before_lst = elements[max(0, i - self.context_length):(i)][::-1]
            context_after_lst = elements[(i + 2):min(len(elements) + 1, i + self.context_length + 1)]

            # have to compute the dictionaries through a for loop so that we can
            # add multiple weights for multiple instances of the word

            context_before_dict = {}
            context_after_dict = {}

            for i in range(len(context_before_lst)):
                context_before_dict.setdefault(context_before_lst[i], 0)
                context_before_dict[context_before_lst[i]] += 1 / (i + 1) # this works because we reversed it above

            for i in range(len(context_after_lst)):
                context_after_dict.setdefault(context_after_lst[i], 0)
                context_after_dict[context_after_lst[i]] += 1 / (i + 1)

            instances.append({
                0: {content_left: 1},
                1: {content_right: 1},
                2: context_before_dict,
                3: context_after_dict,
            })

        # Creating base layer of composite nodes
        for instance in instances:

            # categorization is for finding the best concept to use as a label!
            # this is cool because the label might be useless but the actual
            # instance description is still functional so when we add to the
            # tree, the nodes will be relevant

            cat_res = self.ltm_hierarchy.categorize(instance).get_basic_level()

            inst_dict = instance
            concept_id = self.value_to_id[f"CONCEPT-{cat_res.concept_hash()}"]

            node = ParseNode.create_node(inst_dict, concept_id)

            node.set_parent(self.global_root_node)

            self.nodes.append(node)

        # Creating other layers of composite nodes
        while True:

            best_candidate = None # stores concepts in the form of (score, label, node_left, node_right)

            parentless = self.global_root_node.children
            # TODO one change we can make here is constantly keeping a candidate list and then not
            # worry about recomputing everything -> leaving this here for now but subject to change!

            for i in range(len(parentless) - 1):
                node_left = parentless[i]
                node_right = parentless[i + 1]

                merge_inst = ParseNode.create_merge_instance(node_left, node_right)

                candidate_concept = self.ltm_hierarchy.categorize(merge_inst).get_basic_level()
                candidate_concept_id = self.value_to_id[f"CONCEPT-{candidate_concept.concept_hash()}"]

                # TODO Category Utility may not be the correct score to implement
                if not best_candidate or candidate_concept.category_utility() > best_candidate[0]:
                    best_candidate = (candidate_concept.category_utility(), candidate_concept_id, node_left, node_right)

            if isinstance(end_behavior, (int, float)) and best_candidate[0] < end_behavior:
                break

            best_merge_inst = ParseNode.create_merge_instance(best_candidate[2], best_candidate[3])

            candidate_concept = self.ltm_hierarchy.categorize(best_merge_inst).get_basic_level()
            candidate_concept_id = self.value_to_id[f"CONCEPT-{candidate_concept.concept_hash()}"]

            add_parse_node = ParseNode.create_node(best_merge_inst, candidate_concept_id)

            # changing parentship!
            self.nodes.append(add_parse_node)
            add_parse_node.set_parent(self.global_root_node)
            best_candidate[2].set_parent(add_parse_node)
            best_candidate[3].set_parent(add_parse_node)

            if end_behavior == "converge" and len(self.global_root_node.children) == 1:
                break

        return True

    def get_chunk_instances(self):
        """
        Primary method that returns all available nonterminals as instances
        ready to be passed into the long-term hierarchy
        """

        instances = []

        def dfs_insts(node):
            instances.append(node.get_as_instance())

            for child in node.children:
                dfs_insts(node)

        for child in self.global_root_node.children:
            dfs_insts(child)

        return instances

    def visualize(self, out_base="parse_tree", display_in_colab=True):
        """
        Helper method that visualizes subtrees, with nodes represented as tables
        that relay content and context.
        """
        d3_json = json.dumps(self._tree_to_json())
        html = _build_html(d3_json)
        html_path = f"{out_base}.html"
        png_path  = f"{out_base}.png"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Render to PNG
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(self._html_to_png(html_path, png_path))

        if display_in_colab:
            display(Image(filename=png_path))
        return html_path, png_path

    def _safe_lookup(self, idx):
        return self.id_to_value[idx] if (idx is not None and 0 <= idx < len(self.id_to_value)) else "None"

    def _node_to_dict(self, node, children_getter):
        left_id  = None if not node.content_left  else next(iter(node.content_left.keys()))
        right_id = None if not node.content_right else next(iter(node.content_right.keys()))
        left  =self._safe_lookup(left_id)
        right = self._safe_lookup(right_id)

        def ctx_list(self, ctx):
            if not ctx: return []
            items = sorted(ctx.items(), key=lambda kv: (-kv[1], kv[0]))
            return [{"key": self._safe_lookup(k), "val": float(v)} for k, v in items]

        return {
            "title": node.title,
            "left": left,
            "right": right,
            "before": ctx_list(node.context_before),
            "after":  ctx_list(node.context_after),
            "children": [self._node_to_dict(ch, children_getter) for ch in children_getter(node)]
        }

    def _tree_to_json(self):
        def children_getter(n):
            for ch in getattr(n, "children", []):
                yield (self.nodes[ch] if isinstance(ch, int) else ch)
        return self._node_to_dict(self.global_root_node, children_getter)

    def _build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>Parse Tree</title>
    <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
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
    <div id="tree"></div>
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
    const width  = x1 - x0 + nodeW + 120;
    const height = y1 - y0 + nodeH + 120;

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
    return `
    <div class="node-fo">
        <table><tr><th colspan="2">${{d.title}}</th></tr></table>
        <table>
        <tr><td>Content-Left</td><td>${{d.left}}</td></tr>
        <tr><td>Content-Right</td><td>${{d.right}}</td></tr>
        </table>
        ${{ctxTable(d.before, "Context-Before")}}
        ${{ctxTable(d.after,  "Context-After")}}
    </div>`;
    }}
    </script>
    </body>
    </html>
    """

    async def _html_to_png(html_path, png_path, scale=2):
        browser = await launch(headless=True, args=["--no-sandbox"])
        page = await browser.newPage()
        await page.goto("file://" + os.path.abspath(html_path))

        # wait until tree is rendered
        await page.waitForSelector("#tree svg")

        # get bounding box of the rendered SVG content
        bounding_box = await page.evaluate("""
            () => {
                const svg = document.querySelector('#tree svg');
                const bbox = svg.getBBox();
                return {
                    width: Math.ceil(bbox.x + bbox.width + 20),
                    height: Math.ceil(bbox.y + bbox.height + 20)
                };
            }
        """)

        # set viewport to the exact SVG content size
        await page.setViewport({
            "width": bounding_box['width'],
            "height": bounding_box['height'],
            "deviceScaleFactor": scale
        })

        svg_elem = await page.querySelector("#tree svg")
        await svg_elem.screenshot({"path": png_path, "omitBackground": False})
        await browser.close()